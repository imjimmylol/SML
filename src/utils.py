# src.utils.py
import torch
from typing import Dict, Optional, Tuple
from .env_pack_test import Obs


def modelout_to_dict_test(model_output: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    將模型輸出張量轉換為包含 'action', 'value', 'log_prob' 的字典。
    假設輸入張量的形狀為 (B, 3)，其中每一列分別對應 action, value, log_prob。
    """
    return {
        "consumption": model_output[:, :, 0],
        "delta_savings": model_output[:, :, 1]
    }

def log_state_details(state: Dict[str, Dict[str, torch.Tensor]]):
    """
    詳細印出 state 字典中每個 tensor 的資訊 (shape, dtype, device)。
    """
    print("="*50)
    print("🔎 Logging State Details")
    print("="*50)
    for name, data in state.items():
        tensor = data["value"]
        print(f"🔹 state['{name}']: ")
        print(f"   - Shape:  {tuple(tensor.shape)}")
        print(f"   - Dtype:  {tensor.dtype}")
        print(f"   - Device: {tensor.device}")
    print("="*50 + "\n")

def log_obs_details(obs: Dict[str, Obs]):
    """
    詳細印出 obs 物件中每個 tensor 的資訊 (shape, dtype, device)。
    """
    print("="*50)
    print("🔎 Logging Observation (Obs) Details")
    print("="*50)
    for branch_name, obs_pack in obs.items():
        print(f"--- Branch '{branch_name}' ---")
        
        # Log features
        features_tensor = obs_pack.features
        print(f"🔸 obs['{branch_name}'].features:")
        print(f"   - Shape:  {tuple(features_tensor.shape)}")
        print(f"   - Dtype:  {features_tensor.dtype}")
        print(f"   - Device: {features_tensor.device}")

        # Log condi
        condi_tensor = obs_pack.condi
        print(f"🔸 obs['{branch_name}'].condi:")
        print(f"   - Shape:  {tuple(condi_tensor.shape)}")
        print(f"   - Dtype:  {condi_tensor.dtype}")
        print(f"   - Device: {condi_tensor.device}")

        # Log env_info
        print(f"🔸 obs['{branch_name}'].env_info:")
        if not obs_pack.env_info:
            print("   - (empty)")
        else:
            for info_name, info_tensor in obs_pack.env_info.items():
                print(f"   - '{info_name}':")
                print(f"     - Shape:  {tuple(info_tensor.shape)}")
                print(f"     - Dtype:  {info_tensor.dtype}")
                print(f"     - Device: {info_tensor.device}")
    print("="*50)

def transition_ability_batched(
    ability: torch.Tensor,              # (B, A)
    is_superstar_prev: torch.Tensor,   # (B, A)  bool
    ability_history: torch.Tensor | None,    # (T, B, A) 或 None
    rho_ability: float,
    sigma_ability: float,
    p: float,     # normal -> superstar 的機率
    q: float,     # superstar 留在 superstar 的機率
    ability_bar: float,
    v_min: float,
    v_max: float,
    eps: float = 1e-12,                 # 避免 log(0)
    rng: Optional[torch.Generator] = None,
    zero: bool = False                  # 關閉隨機性
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    向量化、可處理 (B, A) 的版本。
    規則：
      - 對於「原本是 normal」者：以機率 p 升到 superstar
      - 對於「原本是 superstar」者：以機率 q 留在 superstar，否則降回 normal
      - normal 狀態依照 log-AR(1)：log v_t = rho * log v_{t-1} + sigma * eps
      - superstar 狀態：v_t = v_bar * avg_ability，其中 avg_ability 取自 v_history 的 (T, A) 平均，再保留每個 batch 的差異
      - 若 zero=True，關閉所有隨機性：狀態不轉移，AR(1) shock 設為 0
    """
    assert ability.shape == is_superstar_prev.shape, "v_prev 與 is_superstar_prev 形狀需一致"
    device = ability.device
    B, A = ability.shape

    # ===== 1) 狀態轉移 =====
    if zero:
        # 關閉隨機性：狀態保持不變
        u = torch.ones((B, A), device=device)  # 設為 1.0，防止任何轉移
    else:
        u = torch.rand((B, A), device=device)

    # 從 normal 升到 super
    promote = (~is_superstar_prev) & (u < p)
    # superstar 是否留在 superstar
    stay_super = is_superstar_prev & (u < q)

    is_superstar_next = promote | stay_super   # 其餘的就會是 normal

    # ===== 2) 算 superstar 需要用到的平均能力 =====
    if ability_history is not None and ability_history.numel() > 0:
        # v_history: (T, B, A) → 對時間與 agent 取平均，留下每個 batch 的均值 (B, 1)
        avg_per_batch = ability_history.mean(dim=(0, 2), keepdim=True)  # (1, B, 1)
        avg_per_batch = avg_per_batch.squeeze(0)                  # (B, 1)
    else:
        # 若無歷史，就用當期 (B, A) 的 batch 內平均
        avg_per_batch = ability.mean(dim=1, keepdim=True)          # (B, 1)

    # ===== 3) 計算 v_next =====
    ability_next = torch.empty_like(ability)

    # normal 狀態：log-AR(1)
    normal_mask = ~is_superstar_next
    if normal_mask.any():
        if zero:
            # 關閉隨機性：shock 設為 0
            shocks = torch.zeros(ability[normal_mask].shape, device=device)
        else:
            shocks = torch.randn(ability[normal_mask].shape, generator=rng, device=device)
            shocks = shocks.clamp(min=v_min, max=v_max) # normalize shock
        log_ability = rho_ability * torch.log(torch.clamp(ability[normal_mask], min=eps)) + sigma_ability * shocks
        ability_nxt_normal = torch.exp(log_ability)
        ability_next[normal_mask] = torch.clamp(ability_nxt_normal, min=v_min, max=v_max)

    # superstar 狀態：v_bar * (該 batch 的平均能力)
    if is_superstar_next.any():
        # 把 (B,1) 的 batch-均值 broadcast 到 (B,A)
        ability_super = ability_bar * avg_per_batch
        ability_super = ability_super.expand(B, A)
        ability_next[is_superstar_next] = ability_super[is_superstar_next]

    return ability_next, is_superstar_next


def update_ability_history(
    ability_history: torch.Tensor | None,
    ability_next: torch.Tensor,          # (B, A)
    max_len: int | None = None
) -> torch.Tensor:
    """
    動態更新 v_history；若 max_len 有設，僅保留最後 max_len 期。
    形狀：v_history 為 (T, B, A)；v_next 為 (B, A)
    """
    ability_next = ability_next.detach().unsqueeze(0)  # (1, B, A)
    if ability_history is None:
        out = ability_next
    else:
        out = torch.cat([ability_history, ability_next], dim=0)  # (T+1, B, A)
    if max_len is not None and out.shape[0] > max_len:
        out = out[-max_len:]
    return out


def _summarize_tensor(x, name="tensor"):
    x = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
    xf = x.reshape(-1).to(torch.float32)
    isnan = torch.isnan(xf)
    isinf = torch.isinf(xf)
    valid = ~(isnan | isinf)
    xv = xf[valid] if valid.any() else torch.tensor([], dtype=xf.dtype)

    def _safe(fn, default=float('nan')):
        return fn(xv).item() if xv.numel() > 0 else default

    pos = (xv > 0).sum().item()
    neg = (xv < 0).sum().item()
    zer = (xv == 0).sum().item()
    tot = xv.numel()

    def frac(n): return (n / tot) if tot > 0 else float('nan')

    print(f"{name:>12s}: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"{'':12s}  min={_safe(torch.min):.4e}  max={_safe(torch.max):.4e}  mean={_safe(torch.mean):.4e}")
    print(f"{'':12s}  pos={pos} ({frac(pos):.2%})  neg={neg} ({frac(neg):.2%})  zero={zer} ({frac(zer):.2%})")
    print(f"{'':12s}  NaN={isnan.sum().item()}  Inf={isinf.sum().item()}  valid={tot}")