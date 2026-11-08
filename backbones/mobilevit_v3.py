import torch
import torch.nn as nn
from einops import rearrange

# --------------------------------------------------------
# 1. Base Convolutional and CNN Blocks
# --------------------------------------------------------
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def dw_conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    """Depthwise convolution explicitly required by MobileViTv3 paper"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    """Using your original v1 MV2Block for structural parity"""
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# --------------------------------------------------------
# 2. Linear Attention & MobileViTv3 Block
# --------------------------------------------------------
class LinearAttention(nn.Module):
    """O(N) linear attention to replace standard O(N^2) MHSA"""
    def __init__(self, dim, heads=4, dim_head=8, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # x is natively 3D here: (B*P, N, D)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        
        # Linear O(N) attention math
        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        
        out = rearrange(out, 'b h n e -> b n (h e)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                LinearAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))
    def forward(self, x):
        for ln1, attn, ln2, ff in self.layers:
            x = attn(ln1(x)) + x
            x = ff(ln2(x)) + x
        return x

class MobileViTv3Block(nn.Module):
    """The SOTA v3 Fusion block (1x1 pointwise instead of 3x3)"""
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # V3 ARCHITECTURE CHANGE: This is now explicitly depthwise
        self.conv1 = dw_conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_1x1_bn(2 * channel, channel) # v3 fast fusion
        
    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)

        # 1. Capture original spatial dimensions
        B, _, H, W = x.shape
        
        # 2. Fold the patch area (ph, pw) into the Batch dimension
        x = rearrange(x, 'b d (h ph) (w pw) -> (b ph pw) (h w) d', ph=self.ph, pw=self.pw)
        
        # 3. Standard 1D Transformer processing
        x = self.transformer(x)
        
        # 4. Unfold the Sequence back into the 4D Spatial Image
        x = rearrange(x, '(b ph pw) (h w) d -> b d (h ph) (w pw)', 
                      b=B, ph=self.ph, pw=self.pw, h=H//self.ph, w=W//self.pw)

        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x

# --------------------------------------------------------
# 3. The Main MobileViTv3 Backbone (Mapped to v1 layout)
# --------------------------------------------------------
class MobileViTv3(nn.Module):
    def __init__(self, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3] # Transformer depths at each stage

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   
        self.mv2.append(MV2Block(channels[3], channels[4], 1, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTv3Block(dims[0], L[0], channels[5], kernel_size, patch_size=patch_size, mlp_dim=int(dims[0]*2)))
        self.mvit.append(MobileViTv3Block(dims[1], L[1], channels[7], kernel_size, patch_size=patch_size, mlp_dim=int(dims[1]*4)))
        self.mvit.append(MobileViTv3Block(dims[2], L[2], channels[9], kernel_size, patch_size=(1, 1), mlp_dim=int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])


    def forward(self, x):
        multiscale_features = []

        x = self.conv1(x)
        x = self.mv2[0](x)
        multiscale_features.append(x) # stage 1: 56x56

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)     

        x = self.mv2[4](x)
        x = self.mvit[0](x)
        multiscale_features.append(x) # stage 2: 28x28

        x = self.mv2[5](x)
        x = self.mvit[1](x)
        multiscale_features.append(x) # stage 3: 14x14

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        multiscale_features.append(x) # stage 4: 7x7

        return multiscale_features

# --------------------------------------------------------
# 4. Builder Functions
# --------------------------------------------------------
def mobilevit_xxs(image_size=(112, 112)):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViTv3(image_size=image_size, dims=dims, channels=channels, expansion=2)

def mobilevit_xs(image_size=(112, 112)):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViTv3(image_size=image_size, dims=dims, channels=channels)

def mobilevit_s(image_size=(112, 112)):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViTv3(image_size=image_size, dims=dims, channels=channels)

def mobilevit_b(image_size=(112, 112)):
    dims = [192, 256, 384]
    channels = [32, 64, 96, 96, 128, 128, 192, 192, 256, 256, 768]
    return MobileViTv3(image_size=image_size, dims=dims, channels=channels)