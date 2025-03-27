import torch
import torch.nn as nn
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num=8, hidden_size=768, dropout=0.1):
        super().__init__()

        assert hidden_size % head_num == 0, (
            f"hidden_size must be divisible by head_num. "
            f"Got hidden_size={hidden_size} and head_num={head_num}"
        )

        self.head_dim = hidden_size // head_num
        self.head_num = head_num
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        # (b, s, hidden_size)
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # (b, head_num, seq_len, head_dim)
        q_head = Q.view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        k_head = K.view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v_head = V.view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        # (b, head_num, s, s)
        attention_weight = q_head @ k_head.transpose(-2, -1) / math.sqrt(self.head_dim)

        # (b, head_num, s, s)
        if mask is not None:
            assert mask.dim() == 4, f"Expected mask to be 4D, got {mask.dim()}D"
            attention_weight = attention_weight.masked_fill(
                mask, float('-inf')
            )
        
        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.dropout(attention_weight)

        # (b, head_num, s, head_dim) -> (b, s, head_num, head_dim) -> (b, s, head_num*head_dim)
        attention_score = attention_weight @ v_head
        attention_score = attention_score.transpose(1, 2).reshape(batch_size, seq_len, -1)

        output = self.o_proj(attention_score)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=14, embed_dim=768, seq_len=196, dropout=0.1):
        super().__init__()

        # (batch_size, in_channel, img_shape1, img_shape2) -> (b, in*)
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)

        seq_len =  seq_len + 1 # 展平之后的序列长度 + cls_token
        self.position_embedding = nn.Parameter(torch.randn(size=(1, seq_len, embed_dim)).normal_(0.02), requires_grad=True) # "0.02" from BERT
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.shape[0]
        # (1, 1, embed_dim) -> (batch_size, 1, embed_dim)
        batch_class_token = self.cls_token.expand(batch_size, -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.position_embedding 
        x = self.dropout(x)
        return x
    
class MlpBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, 
                 head_num, 
                 embed_dim, 
                 mlp_dim, 
                 attention_dropout, 
                 dropout,
                 ):
        super().__init__()

        # Attention block 
        self.norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attention = MultiHeadAttention(head_num, embed_dim, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MlpBlock(embed_dim, mlp_dim, dropout)
    
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.norm1(input)
        x = self.multihead_attention(x, mask=None)
        x = self.dropout(x)
        x = x + input

        y = self.norm2(x)
        y = self.mlp(y)

        output = x + y

        return output

class Encoder(nn.Module):
    def __init__(self, 
                 head_num, 
                 embed_dim,
                 mlp_dim,
                 attention_dropout,
                 dropout,
                 num_layers):
        super().__init__()

        self.layers = nn.Sequential(
            *[EncoderLayer(head_num, embed_dim, mlp_dim, attention_dropout, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.norm(x)
        
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""
    def __init__(self, 
                 image_size: int,
                 in_channels: int,
                 patch_size: int,
                 head_num: int,
                 embed_dim: int,
                 mlp_dim: int,
                 num_layers,
                 attention_dropout: float = 0.0,
                 dropout: float = 0.0,
                 num_classes: int = 1000,
                 ):
        super().__init__()

        self.image_size = image_size

        seq_len = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, seq_len, dropout)

        self.encoder = Encoder(head_num, embed_dim, mlp_dim, attention_dropout, dropout, num_layers)

        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            nn.init.trunc_normal_(module.weight, std=math.sqrt(1 / fan_in))
    
    def forward(self, x):
        _, _, h, w = x.shape
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")

        x = self.patch_embedding(x)
        x = self.encoder(x)
        cls = self.head(x[:, 0, :]) # 取第一维的cls_token

        return cls
    
'''
MLP_size = 4*embedding_size
ViT-Base: layers=12, embedding_size=768, MLP_size=3072, Attention_heads=12
ViT-Large:  layers=24, embedding_size=1024, MLP_size=4096, Attention_heads=16
ViT-Huge:  layers=32, embedding_size=1280, MLP_size=5120, Attention_heads=16
'''

if __name__ == '__main__':
    vit = VisionTransformer(
        in_channels=3,
        image_size=224,
        patch_size=16,
        num_layers=12,
        head_num=12,
        embed_dim=768,
        mlp_dim=3072,
    )
    print(vit(torch.rand(1, 3, 224, 224)).shape)


    
    
        

