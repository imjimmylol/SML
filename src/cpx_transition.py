# src/cpx_transition.py
import torch
import copy
from typing import Optional, Tuple, Callable
from .env_buffer import StateDict, Actions, Info, _val, _set
from .utils import transition_ability_batched

# Define the function signature type for clarity
TransitionFn = Callable[
    [StateDict, Actions, torch.Generator, Optional[torch.Tensor]],
    Tuple[StateDict, Info]
]

def create_shock_aware_transition(config) -> TransitionFn:
    """
    Factory function that creates a transition function with config baked in.
    """
    shock_config = config.shock

    def _shock_aware_transition(
        state: StateDict,
        actions: Actions,
        rng: torch.Generator,
        v_history: Optional[torch.Tensor]
    ) -> Tuple[StateDict, Info]:

        # 1. Evolve money and savings based on agent actions
        money = _val(state, "moneydisposable")
        sav = _val(state, "savings")
        tax = _val(state, "tax_params")
        B, A = money.shape

        cons = actions.get("consumption", torch.zeros_like(money))
        dsave = actions.get("delta_savings", torch.zeros_like(sav))
        tau_s = tax[:, 0:1].expand(B, A)

        sav_next = (sav + dsave).clamp_min(0.0)
        money_next = (money - cons).clamp_min(0.0)
        money_next = money_next * (1.0 - 0.0 * tau_s)

        # 2. Evolve ability (v) using the custom stochastic process
        v_prev = _val(state, "v")
        is_superstar_prev = _val(state, "is_superstar")

        v_next, is_superstar_next = transition_ability_batched(
            v_prev=v_prev,
            is_superstar_prev=is_superstar_prev,
            v_history=v_history,
            rho_v=shock_config.rho_v,
            sigma_v=shock_config.sigma_v,
            p=shock_config.p,
            q=shock_config.q,
            v_bar=shock_config.v_bar,
            v_min=shock_config.v_min,
            v_max=shock_config.v_max,
        )

        # 3. Assemble the next state
        next_state = copy.deepcopy(state)
        _set(next_state, "moneydisposable", money_next)
        _set(next_state, "savings", sav_next)
        _set(next_state, "v", v_next)
        _set(next_state, "is_superstar", is_superstar_next)

        info: Info = {
            "avg_cons": cons.mean(dim=1),
            "avg_v": v_next.mean(dim=1),
            "superstar_ratio": is_superstar_next.float().mean(dim=1)
        }
        return next_state, info

    return _shock_aware_transition
