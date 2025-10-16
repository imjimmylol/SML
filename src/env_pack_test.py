# env_pack.py
from dataclasses import dataclass
from typing import Dict, Tuple, Literal
import torch

@dataclass
class Obs:
    """打包給 model 的觀測與補充資訊。"""
    features: torch.Tensor   # (B, A, 2A+2)
    condi: torch.Tensor      # (B, A, Z)  稅制條件
    env_info: Dict[str, torch.Tensor]  # e.g. savings_self, superstar

class EconPackEnv:
    """
    只負責 state 與 obs 的環境包裝（無 transition）。
    - reset(B): 產生 state
    - get_obs(state, branch): 回傳 (features, condi, env_info)
    - get_obs_all(state): 同時回傳 v1 / v2 兩組
    """
    def __init__(
        self,
        agents: int,
        tax_params: Dict[str, float],
        v_min: float,
        v_max: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ):
        self.A = int(agents)
        self.Z = len(tax_params)                 # cond_dim 應與模型的 cond_dim 一致（你目前是 5）
        self.tax_ordered = list(tax_params.values())
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.device = torch.device(device)
        self.dtype = dtype

        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        # 與你模型設計相符：state_dim = 2A + 2（2A 來自全體拼接，再加個人 money、v 各 1）

    # ---------- state ----------
    @torch.no_grad()
    def reset(self, B: int) -> Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]]:
        """
        生成一份 batch 狀態，鍵名與你原本 initial_state 完全一致，方便整合。
        注意：初始時只抽一次 v，令 v1 == v2（clone）以便後續能各自更新而互不影響。
        """
        A, dev, dt = self.A, self.device, self.dtype

        # moneydisposable, savings ~ U(0.1, 2.0)
        money = (0.1 + (2.0 - 0.1) * torch.rand((B, A), generator=self.rng, device=dev, dtype=dt))
        sav   = (0.1 + (2.0 - 0.1) * torch.rand((B, A), generator=self.rng, device=dev, dtype=dt))

        # 只抽一次 v0 ~ U(v_min, v_max)，再做 batch-mean normalization
        v0 = (self.v_min + (self.v_max - self.v_min) *
              torch.rand((B, A), generator=self.rng, device=dev, dtype=dt))
        v0 = v0 / v0.mean(dim=1, keepdim=True)

        # 初值：v1、v2 完全相同，但用 clone() 保證未來分開更新時互不連動
        v1 = v0.clone()
        v2 = v0.clone()

        # super star flags（先全 False）
        ss1 = torch.zeros((B, A), device=dev, dtype=torch.bool)
        ss2 = torch.zeros((B, A), device=dev, dtype=torch.bool)

        # 稅制參數 (B, Z) 之後會 broadcast 成 (B, A, Z) 當 condi 使用
        tax = torch.tensor(self.tax_ordered, device=dev, dtype=dt).view(1, self.Z).expand(B, -1)

        # 與你原字典格式對齊（value/shape）
        def pack(x: torch.Tensor):
            return {"value": x, "shape": tuple(x.shape)}

        return {
            "moneydisposable": pack(money),
            "savings":         pack(sav),
            "v1":              pack(v1),
            "v2":              pack(v2),
            "is_superstar_v1": pack(ss1),
            "is_superstar_v2": pack(ss2),
            "tax_params":      pack(tax),
        }

    # ---------- obs (build_inputs) ----------
    def _build_inputs(
        self,
        moneydisposable: torch.Tensor,  # (B, A)
        savings: torch.Tensor,          # (B, A)
        v: torch.Tensor,                # (B, A)
        is_superstar: torch.Tensor | None,  # (B, A) 或 (B, A, 1) 或 scalar
        tax_params: torch.Tensor,       # (B, Z)
        carry_superstar: bool = True,
    ) -> Obs:
        """
        完整復刻你現在的 build_inputs 輸入/輸出介面與 shape。
        """
        B, A = moneydisposable.shape
        # (B, Z) -> (B, A, Z)
        condi = tax_params.unsqueeze(1).expand(-1, A, -1)

        # (B, 2A) -> (B, A, 2A)
        sum_info = torch.cat([moneydisposable, v], dim=1)
        sum_info_rep = sum_info.unsqueeze(1).expand(-1, A, -1)

        # (B, A, 1) x 2
        money_self = moneydisposable.unsqueeze(-1)
        v_self     = v.unsqueeze(-1)

        # features: (B, A, 2A+2)
        features = torch.cat([sum_info_rep, money_self, v_self], dim=2)

        env_info: Dict[str, torch.Tensor] = {}
        env_info["savings_self"] = savings.unsqueeze(-1)  # (B, A, 1)

        if carry_superstar and (is_superstar is not None):
            s = is_superstar
            if s.dim() == 0:
                s = s.to(features).view(1, 1).expand(B, A).unsqueeze(-1)
            elif s.dim() == 2:
                s = s.unsqueeze(-1)
            elif s.dim() == 3:
                pass
            else:
                raise ValueError("is_superstar must be scalar, (B,A), or (B,A,1)")
            env_info["superstar"] = s.to(features.dtype)

        return Obs(features=features, condi=condi, env_info=env_info)

    def get_obs(
        self,
        state: Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]],
        branch: Literal["v1", "v2"],
        carry_superstar: bool = True,
    ) -> Obs:
        """
        取得單一分支 (v1 或 v2) 的觀測。直接拿去丟 model(features, condi) 就行。
        """
        key_v   = "v1" if branch == "v1" else "v2"
        key_ss  = "is_superstar_v1" if branch == "v1" else "is_superstar_v2"
        return self._build_inputs(
            moneydisposable=state["moneydisposable"]["value"],
            savings=state["savings"]["value"],
            v=state[key_v]["value"],
            is_superstar=state[key_ss]["value"],
            tax_params=state["tax_params"]["value"],
            carry_superstar=carry_superstar,
        )

    def get_obs_all(self, state) -> Dict[str, Obs]:
        """同時給出 v1 / v2 兩組 obs。"""
        return {"A": self.get_obs(state, "v1"), "B": self.get_obs(state, "v2")}

    def step(self, *args, **kwargs):
        raise NotImplementedError("此 Env 僅負責打包 state/obs，暫不提供 transition。")
