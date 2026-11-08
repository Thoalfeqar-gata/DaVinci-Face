from typing import Any, Optional, Tuple
from torchvision.ops import MLP
from backbones import backbones
from multitask.face_recognition_subnet import FaceRecognitionEmbeddingSubnet
from utility_scripts import face_recognition_heads
from utility_scripts.pose_estimation_utilities import compute_rotation_matrix_from_ortho6d
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Adapted from FaceXFormer: https://github.com/Kartik-3004/facexformer/blob/main/network/models/facexformer.py
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class MultiScaleProjector(nn.Module):
    def __init__(self, multiscale_feature_dims, output_dim=256):
        super().__init__()
        # Simply project all incoming stages to 256 channels
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim[1], output_dim, kernel_size=1, bias=False),
                LayerNorm2d(output_dim) 
            ) for dim in multiscale_feature_dims
        ])

    def forward(self, multiscale_features):
        return [proj(feat) for proj, feat in zip(self.projectors, multiscale_features)]

class AsymmetricChannelCrossAttention(nn.Module):
    def __init__(self, transformer_embedding_dim=256):
        super().__init__()
        self.transformer_embedding_dim = transformer_embedding_dim
        
        self.q_proj_img = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        self.k_proj_img = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        self.v_proj_task = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        
        self.out_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)

    def forward(self, task_tokens, image_tokens):
        # image_tokens shape: (B, HW, C)
        # task_tokens shape: (B, task_num, C)
        image_task_tokens = torch.cat([image_tokens, task_tokens], dim=1)
        B, N, C = image_task_tokens.shape # N = HW + task_num, C = transformer_embedding_dim

        # 1. The Image and Task tokens dictate the channel correlations
        q = self.q_proj_img(image_task_tokens) 
        k = self.k_proj_img(image_task_tokens) 
        
        # 2. The Task tokens are the values to be dynamically mixed
        v_task = self.v_proj_task(task_tokens) 
        
        # 3. Calculate Image Channel Covariance (C x C)
        # We scale by the sequence length (N) to stabilize the massive sum in the dot product
        scale = N ** -0.5 
        
        # (B, C, N) @ (B, N, C) -> (B, C, C)
        attn_c = (q.transpose(-2, -1) @ k) * scale
        attn_c = attn_c.softmax(dim=-1)
        
        # 4. CROSS-APPLICATION: Image Covariance applied to Task Tokens
        # (B, task_tokens, C) @ (B, C, C) -> (B, task_tokens, C)
        out_task = v_task @ attn_c
        
        return self.out_proj(out_task)


class SpatialCrossAttention(nn.Module):
    def __init__(self, transformer_embedding_dim=256, num_heads=8):
        super().__init__()
        self.transformer_embedding_dim = transformer_embedding_dim
        self.num_heads = num_heads
        self.head_dim = transformer_embedding_dim // num_heads

        self.q_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        self.k_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)
        self.v_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)

        self.out_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)

    def _split_heads(self, x):
        B, N, C = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, image_tokens, image_pe, task_tokens):
            B = image_tokens.size(0)

            q = self._split_heads(self.q_proj(task_tokens)) 
            k = self._split_heads(self.k_proj(image_tokens + image_pe)) # Keys get location
            v = self._split_heads(self.v_proj(image_tokens))            # Values remain pure semantic data

            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(B, -1, self.transformer_embedding_dim)
            
            return self.out_proj(out)

class TaskSelfAttention(nn.Module):
    def __init__(self, transformer_embedding_dim=256, num_heads=8):
        super().__init__()
        self.transformer_embedding_dim = transformer_embedding_dim
        self.num_heads = num_heads
        self.head_dim = transformer_embedding_dim // num_heads

        self.qkv_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim * 3)
        self.out_proj = nn.Linear(transformer_embedding_dim, transformer_embedding_dim)

    def _split_heads(self, x):
        B, N, C = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, task_tokens):
        B = task_tokens.size(0)

        q, k, v = self.qkv_proj(task_tokens).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1, self.transformer_embedding_dim)
        
        # Apply projection to the self-attention output
        return self.out_proj(out)
    
class DaVinciBlockChannelCrossAttention(nn.Module):
    def __init__(self, transformer_embedding_dim=256, num_heads=8, mlp_ratio=8.0, dropout=0.4):
            super().__init__()
            self.spatial_cross_attn = SpatialCrossAttention(transformer_embedding_dim, num_heads)
            self.task_self_attn = TaskSelfAttention(transformer_embedding_dim, num_heads)
            
            self.dynamic_channel_mixer = AsymmetricChannelCrossAttention(transformer_embedding_dim)
            
            self.mlp = nn.Sequential(
                nn.Linear(transformer_embedding_dim, int(transformer_embedding_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(transformer_embedding_dim * mlp_ratio), transformer_embedding_dim),
                nn.Dropout(dropout)
            )
            
            self.norm1 = nn.LayerNorm(transformer_embedding_dim)
            self.norm2 = nn.LayerNorm(transformer_embedding_dim)
            self.norm3 = nn.LayerNorm(transformer_embedding_dim)
            self.norm4 = nn.LayerNorm(transformer_embedding_dim)
        
    def forward(self, image_tokens, image_pe, task_tokens):
        # 1. PRE-NORM Task Self-Attention
        normed_task1 = self.norm1(task_tokens)
        self_attn_output = self.task_self_attn(normed_task1)
        task_tokens = task_tokens + self_attn_output  

        # 2. PRE-NORM Spatial Cross-Attention
        normed_task2 = self.norm2(task_tokens)
        cross_attn_output = self.spatial_cross_attn(image_tokens, image_pe, normed_task2)
        task_tokens = task_tokens + cross_attn_output 

        # 3. PRE-NORM Dynamic Channel Cross Attention
        normed_task3 = self.norm3(task_tokens)
        # Notice we pass the pure image_tokens (no PE) to generate the covariance matrix
        dynamic_out = self.dynamic_channel_mixer(task_tokens=normed_task3, image_tokens=image_tokens[:, 3136:, :]) # Drop the first 3136 image tokens, which correspond to the first stage (56x56). This forces the model to use the later stages (stages 1, 2, and 3) for its dynamic channel mixing, which saves VRAM and also encourages the model to use the higher-resolution features.
        task_tokens = task_tokens + dynamic_out      

        # 4. PRE-NORM Non-Linearity
        normed_task4 = self.norm4(task_tokens)
        mlp_out = self.mlp(normed_task4)
        task_tokens = task_tokens + mlp_out

        return task_tokens
    
class DaVinciBlockBase(nn.Module):
    def __init__(self, transformer_embedding_dim=256, num_heads=8, mlp_ratio=8.0, dropout=0.4, spatial_only = False):
        super().__init__()
        self.spatial_only = spatial_only

        self.spatial_cross_attn = SpatialCrossAttention(transformer_embedding_dim, num_heads)

        if not self.spatial_only:
            self.task_self_attn = TaskSelfAttention(transformer_embedding_dim, num_heads)
        
        self.mlp = nn.Sequential(
            nn.Linear(transformer_embedding_dim, int(transformer_embedding_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(transformer_embedding_dim * mlp_ratio), transformer_embedding_dim),
            nn.Dropout(dropout)
        )
        
        if not self.spatial_only:
            self.norm1 = nn.LayerNorm(transformer_embedding_dim)
        self.norm2 = nn.LayerNorm(transformer_embedding_dim)
        self.norm3 = nn.LayerNorm(transformer_embedding_dim)
        
    
    def forward(self, image_tokens, image_pe, task_tokens):
            if not self.spatial_only:
                # 1. PRE-NORM Task Self-Attention
                normed_task1 = self.norm1(task_tokens)
                self_attn_output = self.task_self_attn(normed_task1)
                task_tokens = task_tokens + self_attn_output  

            # 2. PRE-NORM Spatial Cross-Attention
            normed_task2 = self.norm2(task_tokens)
            cross_attn_output = self.spatial_cross_attn(image_tokens, image_pe, normed_task2)
            task_tokens = task_tokens + cross_attn_output 

            # 3. PRE-NORM MLP
            normed_task3 = self.norm3(task_tokens)
            mlp_output = self.mlp(normed_task3)
            task_tokens = task_tokens + mlp_output      

            return task_tokens

class DaVinciDecoder(nn.Module):
    def __init__(self, num_layers=4, transformer_embedding_dim=256, num_heads=8, mlp_ratio=8.0, dropout=0.4, decoder_type = 'base'):
        super().__init__()

        self.attribute_groups = {
            'Whole' : ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Oval_Face', 'Pale_Skin', 'Male', 'Young'],
            'Hair' : ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'],
            'Eyes' : ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes'],
            'Nose' : ['Big_Nose', 'Pointy_Nose'],
            'Cheek' : ['High_Cheekbones', 'Rosy_Cheeks', 'Wearing_Earrings', 'Sideburns'],
            'Mouth' : ['5_o_Clock_Shadow', 'Big_Lips', 'Mouth_Slightly_Open', 'Mustache', 'Wearing_Lipstick', 'No_Beard', 'Smiling', 'Goatee'],
            'Neck' : ['Wearing_Necklace', 'Wearing_Necktie', 'Double_Chin']
        }
        
        self.tokens_per_task = {
            'emotion_recognition' : 3,
            'age_estimation' : 1,
            'gender_recognition' : 1,
            'race_recognition' : 1,
            'Whole' : 3,          #
            'Hair' : 3,           #
            'Eyes' : 2,           #
            'Nose' : 2,           #     Attribute estimation
            'Cheek' : 2,          #
            'Mouth' : 3,          #
            'Neck' : 2,           #
            'head_pose_estimation' : 1,
            'landmarks_98' : 3,
            'landmarks_68' : 3,
            'landmarks_29_visibility': 3
        }

        self.split_sizes = list(self.tokens_per_task.values())
        self.task_names = list(self.tokens_per_task.keys())

        if decoder_type == 'base':
            self.layers = nn.ModuleList([
                DaVinciBlockBase(transformer_embedding_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
            ])
        elif decoder_type == 'channel_cross_attention':
            self.layers = nn.ModuleList([
                DaVinciBlockChannelCrossAttention(transformer_embedding_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        self.layers.append(DaVinciBlockBase(transformer_embedding_dim, num_heads, mlp_ratio, dropout, spatial_only=True)) # Add a final spatial-only block to ensure strong spatial grounding before the task-specific heads
        
        self.num_tokens = sum(self.tokens_per_task.values()) # emotion, age, gender, race, head pose, landmarks_98, landmarks_68, (landmarks_29 + visibility) + attribute (groups)
        self.task_tokens = nn.Parameter(torch.randn(1, self.num_tokens, transformer_embedding_dim))
        nn.init.trunc_normal_(self.task_tokens, std=0.02)

    def forward(self, image_tokens, image_pe):
        B, HW, C = image_tokens.shape
        H, W = int(HW**0.5), int(HW**0.5)
        
        task_tokens = self.task_tokens.expand(B, -1, -1) 

        for layer in self.layers:
            task_tokens = layer(image_tokens, image_pe, task_tokens)
        
        split_tensors = torch.split(task_tokens, self.split_sizes, dim = 1)
        task_tokens = dict(zip(self.task_names, split_tensors))

        return task_tokens  


class MLP_Head(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout=0.4):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Add the final raw linear projection
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DaVinciFace(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """
            Args:
            num_classes: number of classes for the face recognition task (i.e. number of identities in the training set).
            kwargs: contains all the hyperparameters.
        """
        super(DaVinciFace, self).__init__()

        ##########################
        # Hyperparameters
        # Backbone args
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained_backbone_path = kwargs.get('pretrained_backbone_path')
        self.pretrained_face_recognition_path = kwargs.get('pretrained_face_recognition_subnet_path')
        self.pretrained_margin_head_path = kwargs.get('pretrained_margin_head_path')

        # Face recognition args
        self.head_type = kwargs.get('head_type')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.num_classes = num_classes
        self.margin = kwargs.get('margin')
        self.scale = kwargs.get('scale')
        self.h = kwargs.get('h')
        self.t_alpha = kwargs.get('t_alpha')

        # Decoder args
        self.num_decoder_layers = kwargs.get('num_decoder_layers', 2)
        self.transformer_embedding_dim = kwargs.get('transformer_embedding_dim', 256)
        self.num_heads = kwargs.get('num_heads', 8)
        self.mlp_ratio = kwargs.get('mlp_ratio', 8.0)
        self.dropout = kwargs.get('dropout', 0.4)
        self.decoder_type = kwargs.get('decoder_type', 'base') # 'base' or 'channel_cross_attention'
        ###########################

        # Encoder + face recognition subnet
        self.backbone = backbones.get_backbone(self.backbone_name)
        self.face_recognition_embedding_subnet = FaceRecognitionEmbeddingSubnet(feature_embedding_dim=self.backbone.get_multiscale_feature_dim()[-1][1])
        self.margin_head = face_recognition_heads.build_head(
            head_type=self.head_type,
            embedding_size=self.embedding_dim,
            classnum=self.num_classes,
            m=self.margin,
            h=self.h,
            s=self.scale,
            t_alpha=self.t_alpha
        )

        # Load backbone weights, face recognition subnet weights, and margin head weights if specified
        # Assuming the backbone was pretrained in a compiled model setting, remove the '_orig_mod.' prefix from the state dict keys when loading into a non-compiled model
        if self.pretrained_backbone_path is not None:
            state_dict = torch.load(self.pretrained_backbone_path, map_location='cpu')
            # Always strip '_orig_mod.' prefix if loading into a non-compiled backbone
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict = False)
            print(f'Loaded pretrained backbone from {self.pretrained_backbone_path}.')

        if self.pretrained_face_recognition_path:
            state_dict = torch.load(self.pretrained_face_recognition_path, map_location='cpu')
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.face_recognition_embedding_subnet.load_state_dict(state_dict)
            print(f'Loaded pretrained face recognition subnet from {self.pretrained_face_recognition_path}')

        if self.pretrained_margin_head_path:
            state_dict = torch.load(self.pretrained_margin_head_path, map_location='cpu')
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.margin_head.load_state_dict(state_dict)
            print(f'Loaded pretrained margin head from {self.pretrained_margin_head_path}\n')


        # Decoder
        self.positional_embedding = PositionEmbeddingRandom(num_pos_feats=self.transformer_embedding_dim // 2)
        self.multiscale_projector = MultiScaleProjector(multiscale_feature_dims=self.backbone.get_multiscale_feature_dim(), output_dim=self.transformer_embedding_dim)
        self.decoder = DaVinciDecoder(
            num_layers = self.num_decoder_layers, 
            transformer_embedding_dim = self.transformer_embedding_dim, 
            num_heads = self.num_heads, 
            mlp_ratio = self.mlp_ratio, 
            dropout = self.dropout, 
            decoder_type = self.decoder_type
        )

        self.tokens_per_task = self.decoder.tokens_per_task
        self.attribute_groups = self.decoder.attribute_groups
        self.emotion_recognition_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['emotion_recognition'], self.transformer_embedding_dim * self.tokens_per_task['emotion_recognition'], 7, num_layers=2)
        self.age_estimation_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['age_estimation'], self.transformer_embedding_dim, 102, num_layers=2)
        self.gender_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['gender_recognition'], self.transformer_embedding_dim, 1, num_layers=2)
        self.race_recognition_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['race_recognition'], self.transformer_embedding_dim, 5, num_layers=2)
        self.whole_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Whole'], self.transformer_embedding_dim, len(self.attribute_groups['Whole']), num_layers=2)
        self.hair_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Hair'], self.transformer_embedding_dim, len(self.attribute_groups['Hair']), num_layers=2)
        self.eyes_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Eyes'], self.transformer_embedding_dim, len(self.attribute_groups['Eyes']), num_layers=2)
        self.nose_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Nose'], self.transformer_embedding_dim, len(self.attribute_groups['Nose']), num_layers=2)
        self.cheek_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Cheek'], self.transformer_embedding_dim, len(self.attribute_groups['Cheek']), num_layers=2)
        self.mouth_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Mouth'], self.transformer_embedding_dim, len(self.attribute_groups['Mouth']), num_layers=2)
        self.neck_attribute_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['Neck'], self.transformer_embedding_dim, len(self.attribute_groups['Neck']), num_layers=2)
        self.head_pose_estimation_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['head_pose_estimation'], self.transformer_embedding_dim, 6, num_layers=2)
        self.landmarks_98_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['landmarks_98'], self.transformer_embedding_dim * self.tokens_per_task['landmarks_98'], 98*2, num_layers=2)
        self.landmarks_68_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['landmarks_68'], self.transformer_embedding_dim * self.tokens_per_task['landmarks_68'], 68*2, num_layers=2)
        self.landmarks_29_vis_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['landmarks_29_visibility'], self.transformer_embedding_dim * self.tokens_per_task['landmarks_29_visibility'], 29*2, num_layers=2)
        self.visibility_mlp = MLP_Head(self.transformer_embedding_dim * self.tokens_per_task['landmarks_29_visibility'], self.transformer_embedding_dim * self.tokens_per_task['landmarks_29_visibility'], 29, num_layers=2)


        print('Initialized DaVinciFace with the following configuration:')
        for key, value in kwargs.items():
            print(f'  {key}: {value}')
        

    def forward(self, x):
        multiscale_features = self.backbone(x)
        normalized_embedding, embedding_norm = self.face_recognition_embedding_subnet(multiscale_features)

        # 1. Project all stages to 256 channels (maintaining their native resolutions)
        projected_stages = self.multiscale_projector(multiscale_features)

        image_tokens_list = []
        image_pe_list = []

        # 2. Flatten and generate Positional Encodings per stage
        for stage in projected_stages:
            B, C, H, W = stage.shape
            
            # Flatten stage: (B, C, H, W) -> (B, H*W, C)
            flat_stage = stage.flatten(2).transpose(1, 2)
            image_tokens_list.append(flat_stage)

            # Generate PE for this specific HxW resolution
            pe = self.positional_embedding((H, W)) # (C, H, W)
            pe = pe.flatten(1).transpose(0, 1) # (H*W, C)
            pe = pe.unsqueeze(0).expand(B, -1, -1) # (B, H*W, C)
            image_pe_list.append(pe)

        # 3. Concatenate everything into the massive 4,165-length sequence
        image_tokens = torch.cat(image_tokens_list, dim=1) # (B, 4165, C)
        image_pe = torch.cat(image_pe_list, dim=1)         # (B, 4165, C)

        # 4. Decoder Pass
        task_tokens = self.decoder(image_tokens, image_pe)


        # 5. Heads forward pass
        emotion_recognition_output = self.emotion_recognition_mlp(task_tokens['emotion_recognition'].flatten(1))
        age_estimation_output = self.age_estimation_mlp(task_tokens['age_estimation'].flatten(1))
        gender_output = self.gender_mlp(task_tokens['gender_recognition'].flatten(1))
        race_recognition_output = self.race_recognition_mlp(task_tokens['race_recognition'].flatten(1))
        whole_attribute_output = self.whole_attribute_mlp(task_tokens['Whole'].flatten(1))
        hair_attribute_output = self.hair_attribute_mlp(task_tokens['Hair'].flatten(1))
        eyes_attribute_output = self.eyes_attribute_mlp(task_tokens['Eyes'].flatten(1))
        nose_attribute_output = self.nose_attribute_mlp(task_tokens['Nose'].flatten(1))
        cheek_attribute_output = self.cheek_attribute_mlp(task_tokens['Cheek'].flatten(1))
        mouth_attribute_output = self.mouth_attribute_mlp(task_tokens['Mouth'].flatten(1))
        neck_attribute_output = self.neck_attribute_mlp(task_tokens['Neck'].flatten(1))
        head_pose_estimation_output = self.head_pose_estimation_mlp(task_tokens['head_pose_estimation'].flatten(1))
        landmarks_98_output = self.landmarks_98_mlp(task_tokens['landmarks_98'].flatten(1))
        landmarks_68_output = self.landmarks_68_mlp(task_tokens['landmarks_68'].flatten(1))
        landmarks_29_vis_output = self.landmarks_29_vis_mlp(task_tokens['landmarks_29_visibility'].flatten(1))
        visibility_output = self.visibility_mlp(task_tokens['landmarks_29_visibility'].flatten(1))
        
        attribute_recognition_output = torch.cat([whole_attribute_output, hair_attribute_output, eyes_attribute_output, nose_attribute_output, cheek_attribute_output, mouth_attribute_output, neck_attribute_output], dim=1)

        return (normalized_embedding, embedding_norm), emotion_recognition_output, age_estimation_output, gender_output, race_recognition_output, attribute_recognition_output, head_pose_estimation_output, landmarks_98_output, landmarks_68_output, landmarks_29_vis_output, visibility_output
        
    def get_face_recognition_logits(self, normalized_embedding, embedding_norm, labels):
        return self.margin_head(normalized_embedding, embedding_norm, labels)