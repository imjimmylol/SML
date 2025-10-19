# env_pack.py
from dataclasses import dataclass
from typing import Dict, Tuple, Literal
import torch

@dataclass
class Obs:
    """打包給 model 的觀測與補充資訊。"""
    features: torch.Tensor
    condi: torch.Tensor
    env_info: Dict[str, torch.Tensor] # 不包含進模型的資訊

class EconPackEnv:
    """
    只負責 state 與 obs 的環境包裝（無 transition）。
    - reset(B): 產生一個單一世界 state
    - get_obs(state): 從 state 回傳 (features, condi, env_info)
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
        self.Z = len(tax_params)
        self.tax_ordered = list(tax_params.values())
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.device = torch.device(device)
        self.dtype = dtype

        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

    @torch.no_grad()
    def reset(self, B: int) -> Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]]:
        """
        生成一份 batch 狀態，代表一個單一世界。
        """
        A, dev, dt = self.A, self.device, self.dtype

        money = (0.1 + (2.0 - 0.1) * torch.rand((B, A), generator=self.rng, device=dev, dtype=dt))
        sav = (0.1 + (2.0 - 0.1) * torch.rand((B, A), generator=self.rng, device=dev, dtype=dt))

        v = (self.v_min + (self.v_max - self.v_min) *
             torch.rand((B, A), generator=self.rng, device=dev, dtype=dt))
        v = v / v.mean(dim=1, keepdim=True)

        ss = torch.zeros((B, A), device=dev, dtype=torch.bool)
        tax = torch.tensor(self.tax_ordered, device=dev, dtype=dt).view(1, self.Z).expand(B, -1)

        def pack(x: torch.Tensor):
            return {"value": x, "shape": tuple(x.shape)}

        return {
            "moneydisposable": pack(money),
            "savings": pack(sav),
            "v": pack(v),
            "is_superstar": pack(ss),
            "tax_params": pack(tax),
        }

    def get_obs(
        self,
        state: Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]],
        carry_superstar: bool = True,
    ) -> Obs:
        """
        取得觀測。直接拿去丟 model(features, condi) 就行。
        """
        return self._build_inputs(
            moneydisposable=state["moneydisposable"]["value"],
            savings=state["savings"]["value"],
            v=state["v"]["value"],
            is_superstar=state["is_superstar"]["value"],
            tax_params=state["tax_params"]["value"],
            carry_superstar=carry_superstar,
        )

    def _build_inputs(
        self,
        moneydisposable: torch.Tensor,
        savings: torch.Tensor,
        v: torch.Tensor,
        is_superstar: torch.Tensor | None,
        tax_params: torch.Tensor,
        carry_superstar: bool = True,
    ) -> Obs:
        B, A = moneydisposable.shape
        condi = tax_params.unsqueeze(1).expand(-1, A, -1)

        sum_info = torch.cat([moneydisposable, v], dim=1)
        sum_info_rep = sum_info.unsqueeze(1).expand(-1, A, -1)

        money_self = moneydisposable.unsqueeze(-1)
        v_self = v.unsqueeze(-1)

        features = torch.cat([sum_info_rep, money_self, v_self], dim=2)

        env_info: Dict[str, torch.Tensor] = {}
        env_info["savings_self"] = savings.unsqueeze(-1)

        if carry_superstar and (is_superstar is not None):
            s = is_superstar
            if s.dim() == 0:
                s = s.to(features).view(1, 1).expand(B, A).unsqueeze(-1)
            elif s.dim() == 2:
                s = s.unsqueeze(-1)
            env_info["superstar"] = s.to(features.dtype)

        return Obs(features=features, condi=condi, env_info=env_info)

    def step(self, *args, **kwargs):
        raise NotImplementedError("此 Env 僅負責打包 state/obs，暫不提供 transition。")
