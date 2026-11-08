from typing import Any, Optional, Tuple
from torchvision.ops import MLP
from backbones import backbones
from multitask.face_recognition_subnet import FaceRecognitionEmbeddingSubnet
from utility_scripts import face_recognition_heads
from utility_scripts.pose_estimation_utilities import compute_rotation_matrix_from_ortho6d, matrix_to_euler_angles, draw_axis
from utility_scripts.age_estimation_utilities import get_age_expectation
from torchvision.transforms import v2
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from matplotlib import pyplot as plt
from configs.DaVinciFace.train_davit_t_face_age_gender_race_emotion_attribute_pose_landmarks_visibility_DaVinciFace_channel_cross_attention_glint360k_pretrained_4_layer_decoder import config
import torch, torch.nn as nn, torchvision
import torch.nn.functional as F
import numpy as np
import os
import cv2
    
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
                LayerNorm2d(output_dim) # Your safe ConvNeXt LayerNorm
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

    def forward(self, image_tokens, image_pe, task_tokens, return_attention=False):
            B = image_tokens.size(0)

            q = self._split_heads(self.q_proj(task_tokens)) 
            k = self._split_heads(self.k_proj(image_tokens + image_pe)) # Keys get location
            v = self._split_heads(self.v_proj(image_tokens))            # Values remain pure semantic data

            if return_attention:
                attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                attn_weights = attn.softmax(dim=-1)
                out = attn_weights @ v
            else:
                out = F.scaled_dot_product_attention(q, k, v)
                attn_weights = None

            out = out.transpose(1, 2).reshape(B, -1, self.transformer_embedding_dim)
            
            output = self.out_proj(out)

            if return_attention:
                return output, attn_weights
            
            return output

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
    def __init__(self, transformer_embedding_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.4):
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
        
    def forward(self, image_tokens, image_pe, task_tokens, return_attention=False):
        # 1. PRE-NORM Task Self-Attention
        normed_task1 = self.norm1(task_tokens)
        self_attn_output = self.task_self_attn(normed_task1)
        task_tokens = task_tokens + self_attn_output  

        # 2. PRE-NORM Spatial Cross-Attention
        normed_task2 = self.norm2(task_tokens)
        if return_attention:
            cross_attn_output, spatial_attn = self.spatial_cross_attn(image_tokens, image_pe, normed_task2, return_attention=True)
        else:
            cross_attn_output = self.spatial_cross_attn(image_tokens, image_pe, normed_task2)
            spatial_attn = None
            
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

        if return_attention:
            return task_tokens, spatial_attn

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
        
    
    def forward(self, image_tokens, image_pe, task_tokens, return_attention=False):
            if not self.spatial_only:
                # 1. PRE-NORM Task Self-Attention
                normed_task1 = self.norm1(task_tokens)
                self_attn_output = self.task_self_attn(normed_task1)
                task_tokens = task_tokens + self_attn_output  

            # 2. PRE-NORM Spatial Cross-Attention
            normed_task2 = self.norm2(task_tokens)
            if return_attention:
                cross_attn_output, spatial_attn = self.spatial_cross_attn(image_tokens, image_pe, normed_task2, return_attention=True)
            else:
                cross_attn_output = self.spatial_cross_attn(image_tokens, image_pe, normed_task2)
                spatial_attn = None
            
            task_tokens = task_tokens + cross_attn_output 

            # 3. PRE-NORM MLP
            normed_task3 = self.norm3(task_tokens)
            mlp_output = self.mlp(normed_task3)
            task_tokens = task_tokens + mlp_output      

            if return_attention:
                return task_tokens, spatial_attn

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

    def forward(self, image_tokens, image_pe, return_attention=False):
        B, HW, C = image_tokens.shape
        H, W = int(HW**0.5), int(HW**0.5)
        
        task_tokens = self.task_tokens.expand(B, -1, -1) 

        spatial_attention_maps = []
        for layer in self.layers:
            if return_attention:
                task_tokens, spatial_attn = layer(image_tokens, image_pe, task_tokens, return_attention=True)
                spatial_attention_maps.append(spatial_attn)
            else:
                task_tokens = layer(image_tokens, image_pe, task_tokens)
        
        split_tensors = torch.split(task_tokens, self.split_sizes, dim = 1)
        task_tokens_dict = dict(zip(self.task_names, split_tensors))

        if return_attention:
            return task_tokens_dict, spatial_attention_maps

        return task_tokens_dict  


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
    def __init__(self, **kwargs):
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


        # Load backbone weights, face recognition subnet weights, and margin head weights if specified
        # Assuming the backbone was pretrained in a compiled model setting, remove the '_orig_mod.' prefix from the state dict keys when loading into a non-compiled model
        if self.pretrained_backbone_path is not None:
            state_dict = torch.load(self.pretrained_backbone_path, map_location='cpu')
            # Always strip '_orig_mod.' prefix if loading into a non-compiled backbone
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict)
            print(f'Loaded pretrained backbone from {self.pretrained_backbone_path}.')

        if self.pretrained_face_recognition_path:
            state_dict = torch.load(self.pretrained_face_recognition_path, map_location='cpu')
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.face_recognition_embedding_subnet.load_state_dict(state_dict)
            print(f'Loaded pretrained face recognition subnet from {self.pretrained_face_recognition_path}')

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
        

    def forward(self, x, return_attention=False):
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
        if return_attention:
            task_tokens, spatial_attention_maps = self.decoder(image_tokens, image_pe, return_attention=True)
        else:
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

        results = (
            (normalized_embedding, embedding_norm), 
            emotion_recognition_output, 
            age_estimation_output, 
            gender_output,
            race_recognition_output, 
            attribute_recognition_output, 
            head_pose_estimation_output, 
            landmarks_98_output, 
            landmarks_68_output, 
            landmarks_29_vis_output, 
            visibility_output
        )

        if return_attention:
            return results + (spatial_attention_maps,)
        
        return results

def inference_with_attention(model, image_tensor):
    """
    Helper function to perform inference and return attention maps.
    
    Args:
        model: DaVinciFace model instance.
        image_tensor: Preprocessed image tensor [1, 3, 112, 112].
        
    Returns:
        results: Standard model outputs.
        attention_maps: List of spatial attention maps [Heads, TaskTokens, ImageTokens].
    """
    outputs = model(image_tensor, return_attention=True)
    results = outputs[:11]
    attention_maps = outputs[11] # List of layers
    return results, attention_maps

def prepare_image(image_path = os.path.join('data', 'sample_images','148327.jpg'), image_size=(112, 112)):
    attribute_test_transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize(image_size, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cropping_side_length = int(23 / 112 * image_size[0])  # 23 pixels in the original 112x112 image corresponds to the cropping side length


    rgb_img = cv2.imread(image_path, 1)    
    rgb_img = rgb_img[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, image_size)
    rgb_img= np.float32(rgb_img) / 255.0
        
    input_tensor = torch.reshape(attribute_test_transform(rgb_img), (1, 3, image_size[0], image_size[1])).to('cuda').requires_grad_(True)

    rgb_img = (input_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy() + 1)/2

    return rgb_img, input_tensor

def get_attention_maps_and_predictions(model, input_tensor, split_sizes, image_size = (112, 112)):
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, return_attention=True)
        attention_maps = output[-1]

        per_layer_attention_maps = []
        for i in range(len(attention_maps)):
            attention_maps[i] = attention_maps[i].mean(axis = 1).squeeze(0) # average across all heads
            split_tensors = torch.split(attention_maps[i], split_sizes, dim = 0)
            per_task_attention_maps = []
            for j, split_tensor in enumerate(split_tensors):
                stage_0 = split_tensor[:, 0:56*56].reshape(-1, 56, 56)
                stage_1 = split_tensor[:, 56*56:56*56 + 28*28].reshape(-1, 28, 28)
                stage_2 = split_tensor[:, 56*56 + 28*28:56*56 + 28*28 + 14*14].reshape(-1, 14, 14)
                stage_3 = split_tensor[:, 56*56 + 28*28 + 14*14:56*56 + 28*28 + 14*14 + 7*7].reshape(-1, 7, 7)

                stage_0_resized = F.interpolate(stage_0.unsqueeze(0), size=image_size, mode='bicubic').squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                stage_1_resized = F.interpolate(stage_1.unsqueeze(0), size=image_size, mode='bicubic').squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                stage_2_resized = F.interpolate(stage_2.unsqueeze(0), size=image_size, mode='bicubic').squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                stage_3_resized = F.interpolate(stage_3.unsqueeze(0), size=image_size, mode='bicubic').squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                
                combined_attention_map = (stage_0_resized + stage_1_resized + stage_2_resized + stage_3_resized) / 4.0
                combined_attention_map = combined_attention_map.mean(axis = 2) # average across all tokens for the task to get a single attention map per image
                combined_attention_map_normalized = (combined_attention_map - np.min(combined_attention_map)) / (np.max(combined_attention_map) - np.min(combined_attention_map) + 1e-8)
                per_task_attention_maps.append(combined_attention_map_normalized)
            per_layer_attention_maps.append(per_task_attention_maps)
        
        per_layer_attention_maps = np.array(per_layer_attention_maps) # Layers, Tasks, H, W

    per_task_attention_maps = np.mean(per_layer_attention_maps, axis = 0) # Average across layers
    return per_task_attention_maps, output[:-1]


def get_predictions_from_model_output(model_output):
    
        (normalized_embedding, embedding_norm), emotion_recognition_output, age_estimation_output, gender_output, race_recognition_output, attribute_recognition_output, head_pose_estimation_output, landmarks_98_output, landmarks_68_output, landmarks_29_vis_output, visibility_output = model_output
        
        # emotion_recognition
        emotion_label_translation = {
        0 : 'Neutral',
        1 : 'Happy',
        2 : 'Sad',
        3 : 'Surprised',
        4 : 'Afraid',
        5 : 'Disgusted',
        6 : 'Angry'
        }
        
        emotion_prediction = emotion_label_translation[torch.argmax(emotion_recognition_output, dim = 1).item()]
        
        age_prediction = get_age_expectation(age_estimation_output)
        
        gender_prediction = "Male" if int(torch.sigmoid(gender_output) > 0.5) else "Female"
        
        # race recognition
        race_label_translation = {
            0 : 'White',
            1 : 'Black',
            2 : 'Asian',
            3 : 'Indian',
            4 : 'Other'
        }
        
        race_prediction = race_label_translation[torch.argmax(race_recognition_output, dim=1).item()]
        
        # attribute recognition
        attribute_groups = {
                'Whole' : ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Oval_Face', 'Pale_Skin', 'Male', 'Young'],
                'Hair' : ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'],
                'Eyes' : ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes'],
                'Nose' : ['Big_Nose', 'Pointy_Nose'],
                'Cheek' : ['High_Cheekbones', 'Rosy_Cheeks', 'Wearing_Earrings', 'Sideburns'],
                'Mouth' : ['5_o_Clock_Shadow', 'Big_Lips', 'Mouth_Slightly_Open', 'Mustache', 'Wearing_Lipstick', 'No_Beard', 'Smiling', 'Goatee'],
                'Neck' : ['Wearing_Necklace', 'Wearing_Necktie', 'Double_Chin']
        }

        attribute_list = []
        for group_name in attribute_groups.keys():
            attributes = attribute_groups[group_name]
            attribute_list = attribute_list + attributes

        attribute_list = np.array(attribute_list)
        
        attribute_prediction = attribute_list[(torch.sigmoid(attribute_recognition_output) > 0.5).squeeze(0).cpu().numpy()]

        
        euler_angles = matrix_to_euler_angles(compute_rotation_matrix_from_ortho6d(head_pose_estimation_output))
        roll_prediction = euler_angles[0][0]
        pitch_prediction = euler_angles[0][1]
        yaw_prediction = euler_angles[0][2]
        
        landmark_98_predictions = landmarks_98_output.squeeze(0).detach().cpu().numpy()
        landmark_68_predictions = landmarks_68_output.squeeze(0).detach().cpu().numpy()
        landmark_29_visibility_predictions = landmarks_29_vis_output.squeeze(0).detach().cpu().numpy()
        visibility_output = torch.sigmoid(visibility_output).squeeze(0).detach().cpu().numpy()
        
        return emotion_prediction, age_prediction, gender_prediction, race_prediction, attribute_prediction, (roll_prediction, pitch_prediction, yaw_prediction), landmark_98_predictions, landmark_68_predictions, landmark_29_visibility_predictions, visibility_output
        
def get_attention_visualization_from_attribute_estimation_image(model, input_tensor, split_sizes, cropping_side_length = 23, image_size = (112, 112)):
    per_task_attention_maps_attributes, attribute_predictions = get_attention_maps_and_predictions(model, input_tensor, split_sizes)
    per_task_attention_maps_no_attributes, task_predictions = get_attention_maps_and_predictions(model, torchvision.transforms.Resize(image_size)(input_tensor[:, :, cropping_side_length:-cropping_side_length, cropping_side_length:-cropping_side_length]), split_sizes)

    per_task_attention_maps_attributes = per_task_attention_maps_attributes[4:11] # Only the attribute estimation tasks from the uncropped image
    per_task_attention_maps_final = np.concatenate((per_task_attention_maps_no_attributes[0:4], np.mean(per_task_attention_maps_attributes, axis = 0, keepdims = True), per_task_attention_maps_no_attributes[11:]), axis = 0) # All tasks except attribute estimation
    
    rgb_img = (input_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy() + 1)/2
    cropped_rgb_img = (torchvision.transforms.Resize(image_size)(input_tensor[:, :, cropping_side_length:-cropping_side_length, cropping_side_length:-cropping_side_length]).detach().cpu().squeeze().permute(1, 2, 0).numpy() + 1)/2
    
    visualization_images = []
    for i in range(len(per_task_attention_maps_final)):
        if i != 4: # Skip the attribute estimation task
            visualization_image = show_cam_on_image(cropped_rgb_img, per_task_attention_maps_final[i], use_rgb = True, image_weight = 0.5)
            visualization_images.append(visualization_image)
        else:
            visualization_image = show_cam_on_image(rgb_img, per_task_attention_maps_final[i], use_rgb = True, image_weight = 0.5)
            visualization_images.append(visualization_image)
            
    attribute_predictions = attribute_predictions[5]
    task_predictions = task_predictions[0:5] + (attribute_predictions, ) + task_predictions[6:]
    
    return np.array(visualization_images), get_predictions_from_model_output(task_predictions)


def draw_landmarks_on_image(image, landmarks, visibility = None, color_visible = (0, 255, 0), color_invisible = (255, 0, 0), radius = 1, image_size = (112, 112)):

    for i in range(landmarks.shape[0] // 2):
        x, y = int(landmarks[2 * i] * image_size[0]), int(landmarks[2 * i + 1] * image_size[1])
        x = np.clip(x, 0, image_size[0] - 1)
        y = np.clip(y, 0, image_size[1] - 1)
        
        if visibility is not None:
            if visibility[i] < 0.5:
                cv2.circle(image, (x, y), radius, color_visible, 1)
            else:
                cv2.circle(image, (x, y), radius, color_invisible, 1)
        else:
            cv2.circle(image, (x, y), radius, color_visible, 1)
    return image

def get_landmark_and_visibility_visualization_images(rgb_image, landmark_98_predictions, landmark_68_predictions, landmark_29_visibility_predictions, visibility_output, image_size = (112, 112)):    
    landmark_98_img = draw_landmarks_on_image(rgb_image.copy(), landmark_98_predictions, image_size = image_size)
    landmark_68_img = draw_landmarks_on_image(rgb_image.copy(), landmark_68_predictions, image_size = image_size)
    landmark_29_visibility_img = draw_landmarks_on_image(rgb_image.copy(), landmark_29_visibility_predictions, visibility=visibility_output, image_size = image_size)

    return landmark_98_img, landmark_68_img, landmark_29_visibility_img
    

def get_image_visualization(model, input_tensor, split_sizes, cropping_side_length = 23, figure_name = None, image_size = (112, 112)):
    rgb_image = (input_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy() + 1)/2
    cropped_rgb_image = cv2.resize(rgb_image[cropping_side_length:-cropping_side_length, cropping_side_length:-cropping_side_length, :], image_size)
    visualization_images, predictions = get_attention_visualization_from_attribute_estimation_image(model, input_tensor, split_sizes, cropping_side_length = cropping_side_length, image_size = image_size)
    emotion_prediction, age_prediction, gender_prediction, race_prediction, attribute_prediction, (roll_prediction, pitch_prediction, yaw_prediction), landmark_98_predictions, landmark_68_predictions, landmark_29_visibility_predictions, visibility_output = predictions
    landmark_98_img, landmark_68_img, landmark_29_visibility_img = get_landmark_and_visibility_visualization_images(cropped_rgb_image, landmark_98_predictions, landmark_68_predictions, landmark_29_visibility_predictions, visibility_output, image_size = image_size)
    
    pad = 0
    fontsize = 10
    plt.figure(figsize = (12, 11), dpi=100)

    # Reference Image
    plt.subplot(4, 2, 1)
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title(f'Reference Image', fontsize=fontsize, pad = pad)

    # Emotion Recognition
    plt.subplot(4, 2, 2)
    plt.imshow(visualization_images[0])
    plt.axis('off')
    plt.title(f'Emotion Recognition\n Prediciton: {emotion_prediction}', fontsize=fontsize, pad = pad)

    # Age Estimation
    plt.subplot(4, 4, 5)
    plt.imshow(visualization_images[1])
    plt.axis('off')
    plt.title(f'Age Estimation\n Prediction: {age_prediction[0]:.2f}', fontsize=fontsize, pad = pad)

    # Gender Recognition
    plt.subplot(4, 4, 6)
    plt.imshow(visualization_images[2])
    plt.axis('off')
    plt.title(f'Gender Recognition\n Prediction: {gender_prediction}', fontsize=fontsize, pad = pad)

    # Race Recognition
    plt.subplot(4, 4, 7)
    plt.imshow(visualization_images[3])
    plt.axis('off')
    plt.title(f'Race Recognition\n Prediction: {race_prediction}', fontsize=fontsize, pad = pad)

    # Attribute Estimation
    plt.subplot(4, 4, 8)
    plt.imshow(visualization_images[4])
    plt.axis('off')

    counter = 0
    attribute_text = ""
    for i, attribute in enumerate(attribute_prediction):
        if i % 2 == 0 and i != 0:
            attribute_text += "\n"
        attribute_text += f"{attribute}, "
        counter += 1
        if counter >= 4:  # Limit to 4 attributes for display
            break

    attribute_text += '...'
    plt.title(f'Attribute Estimation\n Prediction: {attribute_text}', fontsize=fontsize, pad = pad)


    # Landmarks 98
    plt.subplot(4, 4, 9)
    plt.imshow(visualization_images[6])
    plt.axis('off')
    plt.title(f'Landmarks 98', fontsize=fontsize, pad = pad)

    # Landmarks 68
    plt.subplot(4, 4, 10)
    plt.imshow(visualization_images[7])
    plt.axis('off')
    plt.title(f'Landmarks 68', fontsize=fontsize, pad = pad)

    # Landmarks 29
    plt.subplot(4, 4, 11)
    plt.imshow(visualization_images[8])
    plt.axis('off')
    plt.title(f'Landmarks 29 and visibility', fontsize=fontsize, pad = pad)

    # Head Pose Estimation
    plt.subplot(4, 4, 12)
    plt.imshow(visualization_images[5])
    plt.axis('off')
    plt.title(f'Head Pose Estimation', fontsize=fontsize, pad = pad)

    # Landmarks 98 visualization
    plt.subplot(4, 4, 13)
    plt.imshow(landmark_98_img)
    plt.axis('off')
    plt.title(f'Landmarks 98', fontsize=fontsize, pad = pad)

    # Landmarks 68 visualization
    plt.subplot(4, 4, 14)
    plt.imshow(landmark_68_img)
    plt.axis('off')
    plt.title(f'Landmarks 68', fontsize=fontsize, pad = pad)

    # Landmarks 29 and visibility visualization
    plt.subplot(4, 4, 15)
    plt.imshow(landmark_29_visibility_img)
    plt.axis('off')
    plt.title(f'Landmarks 29 and visibility', fontsize=fontsize, pad = pad)

    plt.subplot(4, 4, 16)
    pose_estimation_image = draw_axis(cropped_rgb_image, yaw_prediction.cpu().numpy(), pitch_prediction.cpu().numpy(), roll_prediction.cpu().numpy(), size=25)
    plt.imshow(pose_estimation_image)
    plt.axis('off')
    plt.title("Head Pose Estimation", fontsize=fontsize, pad = pad)

    plt.tight_layout()
    plt.show()




def main():
    image_path = os.path.join('data', 'sample_images','1.jpg')  # <-------------- Change this to the path of the image you like from the folder data/sample_images
    
    model = DaVinciFace(**config)
    state_dict = torch.load(os.path.join('data', 'models', 'DaVinci Face', 'model.pth'))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()} # remove '_orig_mod.' prefix if it exists

    new_state_dict = {}
    for k, v in state_dict.items():
        if 'margin' not in k: # exclude margin head weights
            new_state_dict[k] = v
        
    model.load_state_dict( # load the complete model
        new_state_dict
    )
    
    rgb_img, input_tensor = prepare_image(image_path = image_path, image_size=(112, 112))
    
    tokens_per_task = {
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
    split_sizes = list(tokens_per_task.values())
    keys = list(tokens_per_task.keys())

    get_image_visualization(model, input_tensor, split_sizes, cropping_side_length = 18, figure_name = 'visualization.png', image_size = (112, 112))

main()