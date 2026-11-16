from torch import nn

from models.modules.attention import Attention
from models.modules.mlp import MLP


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 mlp_ratio,
                 qkv_bias=False,
                 dropout=0.,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, n_heads, qkv_bias,
                              attn_dropout, proj_dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = norm_layer(dim)
        mlp_hid_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hid_dim, dim, dropout)

    def forward(self, x):

        # # 记录注意力权重
        # attn_out, attn_weights = self.attn(self.norm1(x), return_weights=True)

        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        # if return_attn:
        #     return x, attn_weights
        return x

if __name__ == '__main__':
    import torch

    x = torch.randn(1,197,768)
    model = EncoderBlock(768,12,4.)

    print(model(x).size()) # torch.Size([1,197,768])