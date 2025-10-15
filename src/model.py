import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Tuple

@dataclass
class LastDimCtx:
    orig_shape: torch.Size
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*self.orig_shape[:-1], x.shape[-1])

def flatten_last(x: torch.Tensor) -> tuple[torch.Tensor, LastDimCtx]:
    """把最後一維當作 feature；回傳 (flatten 後張量, 還原用 ctx)。"""
    if x.dim() == 2:
        return x, LastDimCtx(x.shape)
    lead = math.prod(x.shape[:-1])
    return x.reshape(lead, x.shape[-1]), LastDimCtx(x.shape)

def unflatten_last(x: torch.Tensor, ctx: LastDimCtx) -> torch.Tensor:
    """用 ctx 還原 flatten 前的前置維度。"""
    return ctx.restore(x)


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int, identity_init: bool = True, mlp_hidden: int | None = None):
        super().__init__()
        def mlp(out):  # 小工具：一層或兩層 MLP
            if mlp_hidden is None:
                return nn.Linear(cond_dim, out)
            return nn.Sequential(nn.Linear(cond_dim, mlp_hidden), nn.SiLU(), nn.Linear(mlp_hidden, out))

        self.gamma_net = mlp(feature_dim)
        self.beta_net  = mlp(feature_dim)
        self.identity_init = identity_init
        if identity_init and isinstance(self.gamma_net, nn.Linear):
            nn.init.zeros_(self.gamma_net.weight); nn.init.zeros_(self.gamma_net.bias)
            nn.init.zeros_(self.beta_net.weight);  nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_flat, xctx = flatten_last(x)
        c_flat, _    = flatten_last(cond)
        gamma = self.gamma_net(c_flat)
        beta  = self.beta_net(c_flat)
        if self.identity_init:
            out = (1 + gamma) * x_flat + beta
        else:
            out = gamma * x_flat + beta
        return unflatten_last(out, xctx)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film1 = FiLMLayer(cond_dim, hidden_dim)
        self.film2 = FiLMLayer(cond_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.do  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # layer 1
        xf, xctx = flatten_last(x)
        h = self.fc1(xf); h = unflatten_last(h, xctx)
        h = self.film1(h, cond)
        h = self.act(h); h = self.do(h)
        # layer 2
        hf, hctx = flatten_last(h)
        h2 = self.fc2(hf); h2 = unflatten_last(h2, hctx)
        h2 = self.film2(h2, cond)
        # residual + norm
        return self.act(self.norm(h2 + x))

class FiLMResNet2In(nn.Module):
    """
    features: (B, state_dim) 或 (B, N, state_dim)
    condi   : (B, cond_dim)  或 (B, N, cond_dim)
    回傳   : (B, output_dim) 或 (B, N, output_dim)
    """
    def __init__(self, state_dim: int, cond_dim: int, hidden_dim: int = 128,
                 num_res_blocks: int = 2, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.state_dim, self.cond_dim = state_dim, cond_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.initial_film = FiLMLayer(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, cond_dim, dropout)
                                     for _ in range(num_res_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor, condi: torch.Tensor) -> torch.Tensor:
        # encode
        ff, fctx = flatten_last(features)
        h = self.state_encoder(ff)
        h = unflatten_last(h, fctx)
        # FiLM stack
        h = self.initial_film(h, condi)
        for blk in self.blocks:
            h = blk(h, condi)
        # head
        hf, hctx = flatten_last(h)
        out = self.head(hf)
        return unflatten_last(out, hctx)

# # ---------------- 使用方式 ----------------
# if __name__ == "__main__":
#     B, N = 32, 10
#     state_dim = 2*N + 2    # 例如：你的原 state 維度（自由指定）
#     cond_dim  = 5          # 例如：5 個稅參數

#     model = FiLMResNet2In(state_dim=state_dim, cond_dim=cond_dim,
#                           hidden_dim=128, num_res_blocks=3, output_dim=1, dropout=0.1)

#     # 2D 範例（無 agents 軸）
#     x2  = torch.randn(B, state_dim)
#     c2  = torch.randn(B, cond_dim)
#     y2  = model(x2, c2)          # (B, 1)

#     # 3D 範例（有 agents 軸）
#     x3  = torch.randn(B, N, state_dim)
#     c3  = torch.randn(B, N, cond_dim)  # 或者 (B, 1, cond_dim) 可用 broadcast 先擴到 N
#     y3  = model(x3, c3)          # (B, N, 1)
