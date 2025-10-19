from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Literal, Optional, Any
import torch

StateDict = Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]]

@dataclass
class Obs:
    features: torch.Tensor     # (B, A, F)
    condi: torch.Tensor        # (B, A, Z)
    env_info: Dict[str, torch.Tensor]

@dataclass
class Packed:
    obs: Obs
    state_view: Optional[Dict[str, torch.Tensor]]
    meta: Dict[str, Any]

class EconPackEnv:
    def __init__(self, agents: int, tax_params: Dict[str, float],
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 seed: int | None = None):
        self.A = int(agents)
        self.Z = len(tax_params)
        self.tax_keys = list(tax_params.keys())
        self.tax_ordered = [tax_params[k] for k in self.tax_keys]
        self.device = torch.device(device)
        self.dtype = dtype
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

    @torch.no_grad()
    def reset(self, B: int) -> StateDict:
        # 這段可沿用你原本 reset 的 pack 格式
        def pack(x: torch.Tensor): return {"value": x, "shape": tuple(x.shape)}
        A, dt, dev = self.A, self.dtype, self.device
        money = torch.rand((B,A), generator=self.rng, device=dev, dtype=dt)
        savings = torch.rand((B,A), generator=self.rng, device=dev, dtype=dt)
        v = torch.rand((B,A), generator=self.rng, device=dev, dtype=dt)
        v = v / v.mean(dim=1, keepdim=True)
        is_superstar = torch.zeros((B,A), device=dev, dtype=torch.bool)
        tax = torch.tensor(self.tax_ordered, device=dev, dtype=dt).view(1, self.Z).expand(B, -1)
        return {
            "moneydisposable": pack(money),
            "savings": pack(savings),
            "v": pack(v),
            "is_superstar": pack(is_superstar),
            "tax_params": pack(tax),
        }

    # ----- 你原本的打包特徵邏輯（可直接沿用/微調） -----
    def _build_obs(self, state: StateDict, carry_superstar: bool = True) -> Obs:
        money = state["moneydisposable"]["value"]
        savings = state["savings"]["value"]
        v = state["v"]["value"]
        is_superstar = state["is_superstar"]["value"]
        tax = state["tax_params"]["value"]

        B, A = money.shape
        condi = tax.unsqueeze(1).expand(-1, A, -1)         # (B, A, Z)

        sum_info = torch.cat([money, v], dim=1)            # (B, 2A)
        sum_info_rep = sum_info.unsqueeze(1).expand(-1, A, -1)  # (B, A, 2A)
        money_self = money.unsqueeze(-1)                    # (B, A, 1)
        v_self = v.unsqueeze(-1)                            # (B, A, 1)
        features = torch.cat([sum_info_rep, money_self, v_self], dim=2)  # (B, A, 2A+2)

        env_info: Dict[str, torch.Tensor] = {"savings_self": savings.unsqueeze(-1)}
        if carry_superstar:
            s = is_superstar
            if s.dim() == 2:
                s = s.unsqueeze(-1)
            env_info["superstar"] = s.to(features.dtype)

        return Obs(features=features, condi=condi, env_info=env_info)

    # ----- 將「整個環境（或子集）」鏡射給 agent / 後處理 -----
    def get_obs(
        self,
        state: StateDict,
        carry_superstar: bool = True,
        *,
        include_state: Literal["none","history","all"] | Iterable[str] = "none",
        history_keys: Iterable[str] = (),
        clone_state: bool = False,
        detach_state: bool = True,
    ) -> Packed:
        obs = self._build_obs(state, carry_superstar=carry_superstar)

        # 決定要鏡射哪些鍵
        if include_state == "none":
            keys: Tuple[str, ...] = ()
        elif include_state == "all":
            keys = tuple(state.keys())
        elif include_state == "history":
            keys = tuple(history_keys)
        else:
            keys = tuple(include_state)

        if not keys:
            return Packed(obs=obs, state_view=None, meta={"schema":"single-v","tax_keys":self.tax_keys})

        view: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k not in state:
                continue
            t = state[k]["value"]
            if detach_state:
                t = t.detach()
            if clone_state:
                t = t.clone()
            view[k] = t

        return Packed(obs=obs, state_view=view, meta={"schema":"single-v","tax_keys":self.tax_keys})
