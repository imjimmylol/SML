import math
import torch
import torch.nn as nn
from typing import Tuple

# --------- helpers ---------
def _flatten_last(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """
    Make input 2D for Linear layers.
    Returns (flat_x, original_shape). We assume features are in the last dim.
    """
    if x.dim() == 2:
        return x, x.shape
    # merge all leading dims except feature dim
    lead = math.prod(x.shape[:-1])
    return x.reshape(lead, x.shape[-1]), x.shape

def _unflatten_last(x: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """
    Restore the leading dims with new last dim = x.shape[-1].
    """
    return x.reshape(*orig_shape[:-1], x.shape[-1])

# --------- FiLM building blocks (cond 狀態與特徵分離版) ---------
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: y = gamma(cond) * x + beta(cond)
       支援 2D 或 3D 輸入（最後一維為 feature/cond 維度）。
    """
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # flatten -> linear -> reshape back
        x_flat, x_shape = _flatten_last(x)
        cond_flat, cond_shape = _flatten_last(cond)

        gamma = self.gamma_net(cond_flat)  # (B*, feature_dim)
        beta  = self.beta_net(cond_flat)   # (B*, feature_dim)
        out   = gamma * x_flat + beta

        return _unflatten_last(out, x_shape)

class ResidualBlock(nn.Module):
    """Residual MLP block with FiLM conditioning.
       支援 2D / 3D（最後一維隨層內維度變動）。
    """
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1   = nn.Linear(hidden_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.film1 = FiLMLayer(cond_dim, hidden_dim)
        self.film2 = FiLMLayer(cond_dim, hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Keep a residual copy with same shape as x
        residual = x

        # flatten -> linear -> reshape -> FiLM -> act -> drop
        x_flat, x_shape = _flatten_last(x)
        h1 = self.fc1(x_flat)
        h1 = _unflatten_last(h1, x_shape)
        h1 = self.film1(h1, cond)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)

        # second layer
        h1_flat, h1_shape = _flatten_last(h1)
        h2 = self.fc2(h1_flat)
        h2 = _unflatten_last(h2, h1_shape)
        h2 = self.film2(h2, cond)

        out = self.activation(h2 + residual)
        return out

# --------- 兩輸入主網路 ---------
class FiLMResNet2In(nn.Module):
    """
    兩個輸入：
      - features: (B, state_dim) 或 (B, N, state_dim) —— 原本 state 向量
      - condi   : (B, cond_dim)  或 (B, N, cond_dim)  —— 條件/外生稅參數

    模型結構：
      features --(state_encoder)--> hidden
                 hidden --(FiLM with condi)--> hidden
                 -> ResidualBlocks(with FiLM)
                 -> output_head -> y
    """
    def __init__(
        self,
        state_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim  = cond_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.initial_film = FiLMLayer(cond_dim, hidden_dim)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, cond_dim, dropout) for _ in range(num_res_blocks)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor, condi: torch.Tensor) -> torch.Tensor:
        """
        features: (B, state_dim) 或 (B, N, state_dim)
        condi   : (B, cond_dim)  或 (B, N, cond_dim)
        回傳   : (B, output_dim) 或 (B, N, output_dim) 形狀與輸入對齊
        """
        # 1) encode features（支援 2D/3D）
        f_flat, f_shape = _flatten_last(features) 
        print(f_flat.shape, f_shape)                   # (?, state_dim)
        h = self.state_encoder(f_flat)                               # (?, hidden_dim)
        h = _unflatten_last(h, f_shape)                              # (B, [N,] hidden_dim)

        # 2) initial FiLM with condi
        h = self.initial_film(h, condi)

        # 3) residual FiLM blocks
        for blk in self.res_blocks:
            h = blk(h, condi)

        # 4) head
        h_flat, h_shape = _flatten_last(h)
        out = self.output_head(h_flat)
        out = _unflatten_last(out, h_shape)                          # (B, [N,] output_dim)
        return out

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
