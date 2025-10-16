# env_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import torch

StateDict = Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]]
Actions = Dict[str, torch.Tensor]     # 自訂：例如 {"consumption": (B,A), "save": (B,A), ...}
Info    = Dict[str, torch.Tensor]     # step 的回傳資訊（任意）

def _val(s: StateDict, key: str) -> torch.Tensor:
    return s[key]["value"]  # 方便取 value

def _set(s: StateDict, key: str, x: torch.Tensor) -> None:
    s[key]["value"] = x
    s[key]["shape"] = tuple(x.shape)

TransitionFn = Callable[[StateDict, Actions, torch.Generator], Tuple[StateDict, Info]]

@dataclass
class EnvBuffer:
    env: "EconPackEnv"
    B: int
    keep_history: bool = True
    history_keys: Tuple[str, ...] = ("moneydisposable", "savings", "v1", "v2")

    def __post_init__(self):
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.state: StateDict = self.env.reset(self.B)
        self.t: int = 0
        self.traj: list[Dict[str, torch.Tensor]] = []
        return self.state

    def get_obs(self, branch: str = "v1", carry_superstar: bool = True):
        return self.env.get_obs(self.state, branch, carry_superstar=carry_superstar)

    def get_obs_all(self):
        return self.env.get_obs_all(self.state)

    def snapshot(self):
        """（可選）保存當前關鍵欄位，用於回放/訓練。"""
        if not self.keep_history: 
            return
        snap = {k: _val(self.state, k).clone() for k in self.history_keys}
        self.traj.append(snap)

    def step(self, actions: Actions, transition_fn: TransitionFn, *, detach: bool = True):
        """
        由外部傳入 transition 函數來更新狀態。
        - detach=True：常見在模擬/收集資料時斷開梯度，避免圖爆炸。
        """
        if detach:
            # 只斷當前 state 的梯度，不影響 transition_fn 裡的需要梯度的運算（如果有）
            for k in self.state.keys():
                v = _val(self.state, k)
                if torch.is_floating_point(v):
                    _set(self.state, k, v.detach())

        self.snapshot()
        # 交給外部的「動態規則」來產生 next_state
        next_state, info = transition_fn(self.state, actions, self.env.rng)

        # 這裡保留對你原始結構的相容性（value/shape）
        self.state = next_state
        self.t += 1
        return info

    # （可選）便捷方法：直接更新某些欄位
    def update_fields(self, **updates: torch.Tensor):
        for k, x in updates.items():
            if k not in self.state:
                raise KeyError(f"{k} not in state")
            _set(self.state, k, x)


def demo_transition(state: StateDict, actions: Actions, rng: torch.Generator) -> Tuple[StateDict, Info]:
    dev = _val(state, "moneydisposable").device
    dt  = _val(state, "moneydisposable").dtype

    money = _val(state, "moneydisposable")
    sav   = _val(state, "savings")
    v1    = _val(state, "v1")
    v2    = _val(state, "v2")
    tax   = _val(state, "tax_params")        # (B, Z)
    A     = v1.shape[1]
    B     = v1.shape[0]

    # ---- 讀取行為（例如）----
    # 消費/儲蓄行為（示意）：若沒提供則為 0
    cons = actions.get("consumption", torch.zeros_like(money))
    dsave = actions.get("delta_savings", torch.zeros_like(sav))
    # 對 v1/v2 的成長率（示意）：例如模型輸出
    g1 = actions.get("growth_v1", torch.zeros_like(v1))
    g2 = actions.get("growth_v2", torch.zeros_like(v2))

    # ---- 稅制示意：用第一個參數當稅率 τ_s（僅示範，依你定義）----
    # tax: (B, Z) → 擴成 (B, A) 以便逐代理計算
    tau_s = tax[:, 0:1].expand(B, A)

    # ---- 狀態演化（示意）----
    # 更新儲蓄與可支配所得（簡化版，依你實際模型改）
    sav_next   = (sav + dsave).clamp_min(0.0)
    money_next = (money - cons).clamp_min(0.0)

    # 對 v1/v2 套用模型決定的成長（或你的生產/收入方程式）
    v1_next = (v1 * (1.0 + g1)).clamp_min(0.0)
    v2_next = (v2 * (1.0 + g2)).clamp_min(0.0)

    # 超級明星旗標（示意：大於某分位數就點亮）
    q = torch.quantile(v1_next, q=0.99, dim=1, keepdim=True)
    ss1_next = (v1_next >= q)  # (B, A) bool
    # v2 同理
    q2 = torch.quantile(v2_next, q=0.99, dim=1, keepdim=True)
    ss2_next = (v2_next >= q2)

    # 稅收對下一期 money 的影響（示意）
    money_next = money_next * (1.0 - 0.0 * tau_s)  # 這行僅示意，實際請按你的模型來

    # ---- 回填（保持 value/shape 結構）----
    next_state = {k: {"value": _val(state, k), "shape": state[k]["shape"]} for k in state.keys()}
    # 必要欄位覆寫
    next_state["moneydisposable"]["value"] = money_next
    next_state["moneydisposable"]["shape"] = tuple(money_next.shape)

    next_state["savings"]["value"] = sav_next
    next_state["savings"]["shape"] = tuple(sav_next.shape)

    next_state["v1"]["value"] = v1_next
    next_state["v1"]["shape"] = tuple(v1_next.shape)

    next_state["v2"]["value"] = v2_next
    next_state["v2"]["shape"] = tuple(v2_next.shape)

    next_state["is_superstar_v1"]["value"] = ss1_next.to(torch.bool)
    next_state["is_superstar_v1"]["shape"] = tuple(ss1_next.shape)

    next_state["is_superstar_v2"]["value"] = ss2_next.to(torch.bool)
    next_state["is_superstar_v2"]["shape"] = tuple(ss2_next.shape)

    info: Info = {
        "avg_cons": cons.mean(dim=1),         # 例：紀錄統計
        "avg_v1":   v1_next.mean(dim=1),
        "avg_v2":   v2_next.mean(dim=1),
    }
    return next_state, info

