import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor

''' Embedded Patches '''
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

''' Norm '''
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

''' Multi-Heat Attention '''
from einops import rearrange
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output

''' FeedForward '''
class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

''' Residual '''
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

''' Vision Transformer '''
from einops import repeat

class ViT1(nn.Module):
    def __init__(self, ch=3, img_size=64, patch_size=16, emb_dim=32,
                n_layers=6, out_dim=2, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim=emb_dim*2, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.to_cls_token = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, img, return_features=False):
        x = self.patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        for layer in self.layers:
            x = layer(x)

        if return_features:
            return x[:, 0, :]  # 마지막 트랜스포머 레이어의 출력 (batch_size, 17, 32)

        x = self.to_cls_token(x[:, 0])  # Identity 레이어 사용
        head_x = self.head(x)  # (batch_size, 2)
        return x, head_x


class ViT2(nn.Module):
    def __init__(self, emb_dim=32, time_step=9, num_classes=2, dropout=0.1, n_layers=6, heads=2):
        super(ViT2, self).__init__()

        # Attributes
        self.emb_dim = emb_dim
        self.time_step = time_step

        # Learnable params
        self.pos_embedding = nn.Parameter(torch.randn(1, time_step + 1, emb_dim))  # time_step = num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim=emb_dim * 2, dropout=dropout))))
            self.layers.append(transformer_block)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.to_cls_token = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, time_step, emb_dim]
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, time_step + 1, emb_dim]
        x += self.pos_embedding[:, :(self.time_step + 1)]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.to_cls_token(x[:, 0])
        return x, self.head(x)  # [batch_size, emb_dim], [batch_size, num_classes]


class ViT3(nn.Module):
    def __init__(self, emb_dim=32, time_step=49, num_classes=10, dropout=0.1, n_layers=6, heads=2):
        super(ViT3, self).__init__()

        # Attributes
        self.emb_dim = emb_dim
        self.time_step = time_step

        # Learnable params
        self.pos_embedding = nn.Parameter(torch.randn(1, time_step + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim=emb_dim * 2, dropout=dropout))))
            self.layers.append(transformer_block)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.to_cls_token = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, time_step, emb_dim]
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, time_step + 1, emb_dim]
        x += self.pos_embedding[:, :(self.time_step + 1)]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.to_cls_token(x[:, 0])  # [batch_size, emb_dim]
        return x, self.head(x)  # [batch_size, num_classes]
