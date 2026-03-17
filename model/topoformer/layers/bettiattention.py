import torch
from torch import nn

class TopoEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 topo_num_heads,
                 dim_feedforward,
                 dropout = 0.0, **kwargs):
        super().__init__()
        self.k_w = nn.Linear(input_size, input_size)
        self.q_w = nn.Linear(input_size, input_size)
        self.v_w = nn.Linear(input_size, input_size)
        self.self_attn = torch.nn.MultiheadAttention(input_size, topo_num_heads, batch_first= True)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_size)
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.k_w(x)
        q = self.q_w(x)
        v = self.v_w(x)
        attn_out, _ = self.self_attn(k, q, v)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        linear_out = self.mlp(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x

class BettiAttention(nn.Module):
    def __init__(self,
                 output_feature_dim: int,
                 topo_num_heads = 1,
                 embedding_dim: int = 100,
                 input_size: int = 3*625,
                 num_encoder_layers = 3,
                 **kwargs):
        super().__init__()
        self.mlp_embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.ReLU(inplace = True),
            nn.Linear(embedding_dim, output_feature_dim),
        )
        self.encoder_layers = nn.ModuleList([TopoEncoder(input_size = output_feature_dim,
                                                         topo_num_heads = topo_num_heads,
                                                         dim_feedforward = output_feature_dim,
                                                         **kwargs) for _ in range(num_encoder_layers)])

    def forward(self, x):
        x = self.mlp_embedding(x)
        for l in self.encoder_layers:
            x = l(x)
        return x
