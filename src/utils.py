import torch
from typing import Tuple


def transition_ability_batched(
    v_prev: torch.Tensor,              # (B, A)
    is_superstar_prev: torch.Tensor,   # (B, A)  bool
    v_history: torch.Tensor | None,    # (T, B, A) 或 None
    rho_v: float,
    sigma_v: float,
    p: float,     # normal -> superstar 的機率
    q: float,     # superstar 留在 superstar 的機率
    v_bar: float,
    v_min: float,
    v_max: float,
    eps: float = 1e-12                 # 避免 log(0)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    向量化、可處理 (B, A) 的版本。
    規則：
      - 對於「原本是 normal」者：以機率 p 升到 superstar
      - 對於「原本是 superstar」者：以機率 q 留在 superstar，否則降回 normal
      - normal 狀態依照 log-AR(1)：log v_t = rho * log v_{t-1} + sigma * eps
      - superstar 狀態：v_t = v_bar * avg_ability，其中 avg_ability 取自 v_history 的 (T, A) 平均，再保留每個 batch 的差異
    """
    assert v_prev.shape == is_superstar_prev.shape, "v_prev 與 is_superstar_prev 形狀需一致"
    device = v_prev.device
    B, A = v_prev.shape

    # ===== 1) 狀態轉移 =====
    u = torch.rand((B, A), device=device)

    # 從 normal 升到 super
    promote = (~is_superstar_prev) & (u < p)
    # superstar 是否留在 superstar
    stay_super = is_superstar_prev & (u < q)

    is_superstar_next = promote | stay_super   # 其餘的就會是 normal

    # ===== 2) 算 superstar 需要用到的平均能力 =====
    if v_history is not None and v_history.numel() > 0:
        # v_history: (T, B, A) → 對時間與 agent 取平均，留下每個 batch 的均值 (B, 1)
        avg_per_batch = v_history.mean(dim=(0, 2), keepdim=True)  # (1, B, 1)
        avg_per_batch = avg_per_batch.squeeze(0)                  # (B, 1)
    else:
        # 若無歷史，就用當期 (B, A) 的 batch 內平均
        avg_per_batch = v_prev.mean(dim=1, keepdim=True)          # (B, 1)

    # ===== 3) 計算 v_next =====
    v_next = torch.empty_like(v_prev)

    # normal 狀態：log-AR(1)
    normal_mask = ~is_superstar_next
    if normal_mask.any():
        shocks = torch.randn_like(v_prev[normal_mask], device=device)
        log_v = rho_v * torch.log(torch.clamp(v_prev[normal_mask], min=eps)) + sigma_v * shocks
        v_nxt_normal = torch.exp(log_v)
        v_next[normal_mask] = torch.clamp(v_nxt_normal, min=v_min, max=v_max)

    # superstar 狀態：v_bar * (該 batch 的平均能力)
    if is_superstar_next.any():
        # 把 (B,1) 的 batch-均值 broadcast 到 (B,A)
        v_super = v_bar * avg_per_batch
        v_super = v_super.expand(B, A)
        v_next[is_superstar_next] = v_super[is_superstar_next]

    return v_next, is_superstar_next


def update_v_history(
    v_history: torch.Tensor | None,
    v_next: torch.Tensor,          # (B, A)
    max_len: int | None = None
) -> torch.Tensor:
    """
    動態更新 v_history；若 max_len 有設，僅保留最後 max_len 期。
    形狀：v_history 為 (T, B, A)；v_next 為 (B, A)
    """
    v_next = v_next.detach().unsqueeze(0)  # (1, B, A)
    if v_history is None:
        out = v_next
    else:
        out = torch.cat([v_history, v_next], dim=0)  # (T+1, B, A)
    if max_len is not None and out.shape[0] > max_len:
        out = out[-max_len:]
    return out
