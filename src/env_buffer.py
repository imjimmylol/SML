# env_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any, Literal
import torch
import copy
from .utils import update_v_history

StateDict = Dict[str, Dict[str, torch.Tensor | Tuple[int, ...]]]
Actions = Dict[str, torch.Tensor]
Info = Dict[str, torch.Tensor]

def _val(s: StateDict, key: str) -> torch.Tensor:
    return s[key]["value"]

def _set(s: StateDict, key: str, x: torch.Tensor) -> None:
    s[key] = {"value": x, "shape": tuple(x.shape)}

# The transition function now also receives the history of v
TransitionFnWithHistory = Callable[
    [StateDict, Actions, torch.Generator, Optional[torch.Tensor]],
    Tuple[StateDict, Info]
]

@dataclass
class EnvBuffer:
    env: "EconPackEnv"
    B: int
    keep_history: bool = True
    history_keys: Tuple[str, ...] = ("moneydisposable", "savings", "v")
    v_history_max_len: int = 100

    # 新增：可選的固定 seed，用來可重現 A/B 的獨立隨機流
    seed: Optional[int] = None

    # 這兩個是每個 branch 的獨立 RNG
    rng_A: Optional[torch.Generator] = None
    rng_B: Optional[torch.Generator] = None



    def __post_init__(self):
        self.reset()

    def _init_branch_rngs(self):
        dev = self.env.device
        self.rng_A = torch.Generator(device=dev)
        self.rng_B = torch.Generator(device=dev)
        # 用一個基底種子推導出兩條獨立流；若未提供，就用 env.rng 的 initial_seed
        base = self.seed if self.seed is not None else self.env.rng.initial_seed()
        # 兩個不同的混洗（64-bit XOR 常數）；保持可重現但互不相同
        self.rng_A.manual_seed((base ^ 0x9E3779B185EBCA87) & 0xFFFFFFFFFFFFFFFF)
        self.rng_B.manual_seed((base ^ 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF)

    @torch.no_grad()
    def reset(self):
        initial_state = self.env.reset(self.B)
        self.state_A = initial_state
        self.state_B = copy.deepcopy(initial_state)
        self.t = 0
        self.traj_A, self.traj_B = [], []
        self._init_branch_rngs()
        # 先把 t=0 的 v 放進歷史
        v0_A = self.state_A["v"]["value"]
        v0_B = self.state_B["v"]["value"]
        self.v_history_A = update_v_history(None, v0_A, self.v_history_max_len)
        self.v_history_B = update_v_history(None, v0_B, self.v_history_max_len)
        return self.state_A, self.state_B

    def get_obs(
        self,
        branch: Literal["A","B"],
        carry_superstar: bool = True,
        *,
        mirror: Literal["none","history","all"] = "history",
        clone_state: bool = False,
        detach_state: bool = True,
    ):
        state = self.state_A if branch == "A" else self.state_B
        # 將 history_keys 傳給 env，實現你要的「連動」
        packed = self.env.get_obs(
            state,
            carry_superstar=carry_superstar,
            include_state=mirror,
            history_keys=self.history_keys,
            clone_state=clone_state,
            detach_state=detach_state,
        )
        return packed

    def snapshot(self):
        if not self.keep_history:
            return
        snap_A = {k: _val(self.state_A, k).clone() for k in self.history_keys}
        snap_B = {k: _val(self.state_B, k).clone() for k in self.history_keys}
        self.traj_A.append(snap_A)
        self.traj_B.append(snap_B)

    def step(
        self,
        actions_A: Actions,
        actions_B: Actions,
        transition_fn: TransitionFnWithHistory,
        *,
        detach: bool = True
    ) -> Dict[str, Info]:
        if detach:
            for state in [self.state_A, self.state_B]:
                for k in state.keys():
                    v = _val(state, k)
                    if torch.is_floating_point(v):
                        _set(state, k, v.detach())

        self.snapshot()

        # Pass the corresponding v_history to the transition function
        next_state_A, info_A = transition_fn(self.state_A, actions_A, self.rng_A, self.v_history_A)
        next_state_B, info_B = transition_fn(self.state_B, actions_B, self.rng_B, self.v_history_B)

        self.state_A = next_state_A
        self.state_B = next_state_B
        
        # Update the v_history for each world
        self.v_history_A = update_v_history(self.v_history_A, _val(self.state_A, "v"), self.v_history_max_len)
        self.v_history_B = update_v_history(self.v_history_B, _val(self.state_B, "v"), self.v_history_max_len)

        self.t += 1
        return {"A": info_A, "B": info_B}

    def update_state_value(self, branch: Literal["A", "B"], key: str, value: torch.Tensor):
        """Convenience method to update a single value in one of the states."""
        state = self.state_A if branch == "A" else self.state_B
        if key not in state:
            raise KeyError(f"Key '{key}' not in state.")
        _set(state, key, value)

"""
# This is a demonstration of a simple transition function.
# It was the original `demo_transition`.
def test(state: StateDict, actions: Actions, rng: torch.Generator) -> Tuple[StateDict, Info]:
    money = _val(state, "moneydisposable")
    sav = _val(state, "savings")
    v = _val(state, "v")
    tax = _val(state, "tax_params")
    B, A = v.shape

    cons = actions.get("consumption", torch.zeros_like(money))
    dsave = actions.get("delta_savings", torch.zeros_like(sav))
    g_v = actions.get("growth_v", torch.zeros_like(v))

    tau_s = tax[:, 0:1].expand(B, A)

    sav_next = (sav + dsave).clamp_min(0.0)
    money_next = (money - cons).clamp_min(0.0)
    v_next = (v * (1.0 + g_v)).clamp_min(0.0)

    q = torch.quantile(v_next, q=0.99, dim=1, keepdim=True)
    ss_next = (v_next >= q)

    money_next = money_next * (1.0 - 0.0 * tau_s)

    next_state = copy.deepcopy(state)
    _set(next_state, "moneydisposable", money_next)
    _set(next_state, "savings", sav_next)
    _set(next_state, "v", v_next)
    _set(next_state, "is_superstar", ss_next.to(torch.bool))

    info: Info = {
        "avg_cons": cons.mean(dim=1),
        "avg_v": v_next.mean(dim=1),
    }
    return next_state, info
"""

