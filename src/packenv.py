from __future__ import annotations
import torch 
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .normalizer import RunningMeanStd
from .configloader import load_config

class EconVecEnv:
    """
    Vectorized economic environment with B parallel worlds and A agents per world.
    Designed for *continuing* tasks (no episodes). All math is torch-differentiable
    except for randomness in shock transitions.
    """

    def __init__(self, agents: int, config: SimpleNamespace, device=None, dtype=torch.float32,
                 normalize_obs: bool = True, seed: int | None = None):

        self.cfg = config  # SimpleNamespace from load_config()
        self.A = int(agents if agents is not None else self.cfg.training.agents)

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        # RNG for env randomness (init/shocks)
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        # --- structural params ---
        self.theta = float(getattr(self.cfg.milf_inputs, "theta", 1.0))
        self.beta  = float(getattr(self.cfg.milf_inputs, "beta",  0.975))
        self.Aprod = float(getattr(self.cfg.milf_inputs, "A",     1.0))
        self.alpha = float(getattr(self.cfg.milf_inputs, "alpha", 0.33))
        self.gamma = float(getattr(self.cfg.milf_inputs, "gamma", 2.0))
        self.delta = float(getattr(self.cfg.milf_inputs, "delta", 0.06))

        # --- shock params (load_config has v_min/v_max) ---
        self.rho_v   = float(self.cfg.shock.rho_v)
        self.sigma_v = float(self.cfg.shock.sigma_v)
        self.v_min   = float(self.cfg.shock.v_min)
        self.v_max   = float(self.cfg.shock.v_max)
        self.v_bar   = float(getattr(self.cfg.shock, "v_bar", 1.5))
        self.p       = float(getattr(self.cfg.shock, "p", 2.2e-6))  # enter superstar
        self.q       = float(getattr(self.cfg.shock, "q", 0.99))    # stay superstar

        # --- tax params ---
        tp = self.cfg.tax_params
        self.tax_params = SimpleNamespace(
            tax_consumption=float(tp.tax_consumption),
            tax_income=float(tp.tax_income),
            income_tax_elasticity=float(tp.income_tax_elasticity),
            saving_tax_elasticity=float(tp.saving_tax_elasticity),
            tax_saving=float(tp.tax_saving),
        )
        self.Z = 5  # condi dim = 5 tax params

        # TODO
        # --- obs normalizer (separate stats for features / condi) ---
        self.normalize_obs = bool(normalize_obs)
        if self.normalize_obs:
            self.rms_features = RunningMeanStd(shape=(2 * self.A + 2))
            self.rms_condi    = RunningMeanStd(shape=(self.Z,))
        # --- internal states (set in reset) ---
        
        self.state = None
        # 可供你兩條路徑各自維護歷史（如果需要）
        self.v_history_A, self.v_history_B = None, None
        self.t = 0

        # 若設 horizon<=0 就當 continuing（done 全 False）
        self.horizon = int(getattr(self.cfg.training, "episode_len", 0))

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _mean_across_agents(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)  # (B,1)

    def _compute_prices(self, K: torch.Tensor, v: torch.Tensor, h: torch.Tensor):
        """
        Aggregate K, effective labor L, Cobb-Douglas Y = A K^α (L)^(1-α).
        Wage: w = (1-α) A (K/L)^α ; Return on capital (gross MPK): r = α A (K/L)^(α-1)
        Returns broadcasted wage, ret as (B,A).
        """
        k_over_l = (self._mean_across_agents(K) / (self._mean_across_agents(h * v).clamp_min(1e-8))).clamp_min(1e-12)
        wage = self.Aprod * (1.0 - self.alpha) * (k_over_l ** self.alpha)          # (B,1)
        ret  = self.Aprod * self.alpha * (k_over_l ** (self.alpha - 1.0))          # (B,1)
        return wage.expand(-1, self.A), ret.expand(-1, self.A)

    @staticmethod
    def _taxfunc(ibt: torch.Tensor, abt: torch.Tensor, tp: SimpleNamespace):
        """
        Placeholder elastic tax schedule; replace with your exact formulas if needed.
        """
        it = ibt - (1 - tp.tax_income) * (ibt ** (1 - tp.income_tax_elasticity)) / (
            1 - tp.income_tax_elasticity + 1e-8
        )
        at = abt - ((1 - tp.tax_saving) / (1 - tp.saving_tax_elasticity + 1e-8)) * (
            abt ** (1 - tp.saving_tax_elasticity)
        )
        return it, at

    def _calculate_money_disposable(self, wage, ret, v, h, k_prev, is_init=False):
        """
        money = after-tax income from labor (w*h*v) plus capital income ((1-δ+r)k_prev)
        """
        if is_init:
            ibt = wage * h * v
        else:
            ibt = wage * h * v + (1.0 - self.delta + ret) * k_prev
        it, at = self._taxfunc(ibt, k_prev, self.tax_params)
        money = it + at
        return money, ibt

    @staticmethod
    def _output_transform(a: torch.Tensor, money: torch.Tensor):
        c = money * (1 - a)
        s = money * a
        return c, s

    def _transition_ability_batched(self, v_prev, is_super_prev, v_hist, eps_v=None):
        """
        Log-AR(1) for ability, plus superstar regime (enter prob p, stay prob q).
        Returns v_next, is_super_next, v_hist_next.
        """
        B, A = v_prev.shape

        # superstar switching
        u = torch.rand((B, A), device=v_prev.device, generator=self.rng)
        stay_super  = is_super_prev & (u < self.q)
        enter_super = (~is_super_prev) & (u < self.p)
        super_next  = (stay_super | enter_super)

        # innovation
        if eps_v is None:
            eps_v = torch.randn((B, A), device=v_prev.device, generator=self.rng)
        log_vn = self.rho_v * v_prev.clamp_min(1e-12).log() + self.sigma_v * eps_v
        v_next = log_vn.exp().clamp(self.v_min, self.v_max)

        # superstar level pegged to global average of history (or current mean if no history)
        if v_hist is not None and v_hist.numel() > 0:
            avg_ability = v_hist.mean(dim=(0, 1))  # scalar or (A,) but broadcastable
        else:
            avg_ability = v_prev.mean()
        v_next = torch.where(super_next, self.v_bar * avg_ability.expand_as(v_next), v_next)

        # maintain short history to avoid OOM
        if v_hist is None:
            v_hist_next = v_next.unsqueeze(0)  # (T=1, B, A)
        else:
            v_hist_next = torch.cat([v_hist, v_next.unsqueeze(0)], dim=0)
            if v_hist_next.shape[0] > 64:
                v_hist_next = v_hist_next[-64:]

        return v_next, super_next, v_hist_next

    # ----------------------------------------------------------------
    # API
    # ----------------------------------------------------------------
    @torch.no_grad()
    def reset(self, batch_size: int, state: dict | None = None, init_mode: str = "random"):
        """
        For *continuing* tasks you typically call reset() ONCE at the very beginning.
        Returns: (obs, state, info)
        obs: {'features': (B,A,2A+2), 'condi': (B,A,Z)}
        """
        B, A = int(batch_size), self.A

        if state is None:
            if init_mode != "random":
                raise ValueError(f"Unknown init_mode: {init_mode}")

            money = torch.empty(B, A, device=self.device, dtype=self.dtype).uniform_(0.1, 2.0, generator=self.rng)
            savings = torch.empty(B, A, device=self.device, dtype=self.dtype).uniform_(0.1, 2.0, generator=self.rng)
            v = torch.empty(B, A, device=self.device, dtype=self.dtype).uniform_(self.v_min, self.v_max, generator=self.rng)
            v = v / v.mean(dim=1, keepdim=True)  # normalize each world's mean ability to 1
            is_super = torch.zeros(B, A, device=self.device, dtype=torch.bool)
        else:
            money     = state["moneydisposable"]
            savings   = state["savings"]
            v         = state["v"]
            is_super  = state["is_superstar"]

        self.state = {
            "moneydisposable": money,
            "savings": savings,
            "v": v,
            "is_superstar": is_super,
            "a": torch.zeros_like(savings),  # placeholder for last period a
        }
        self.v_history_A, self.v_history_B = None, None
        self.t = 0

        # here bug
        obs = self.build_obs(self.state)
        info = {"t": self.t}
        return obs, self.state, info

    def build_obs(self, state: dict) -> dict:
        """
        Build observation for policy/net:
        features: concat[ money_all_agents , v_all_agents , money_self , v_self ] -> (B,A,2A+2)
        condi:    tax parameter vector repeated across agents -> (B,A,Z)
        """
        money = state["moneydisposable"]  # (B,A)
        v     = state["v"]                # (B,A)
        B, A  = money.shape

        # condi = (B,A,Z)
        tp_vec = torch.tensor(
            [self.tax_params.tax_consumption,
             self.tax_params.tax_income,
             self.tax_params.income_tax_elasticity,
             self.tax_params.saving_tax_elasticity,
             self.tax_params.tax_saving],
            device=self.device, dtype=self.dtype
        ).expand(B, self.Z)
        condi = tp_vec.unsqueeze(1).expand(-1, A, -1)

        # features = (B,A,2A+2)
        sum_info = torch.cat([money, v], dim=1)            # (B,2A)
        sum_rep  = sum_info.unsqueeze(1).expand(-1, A, -1) # (B,A,2A)
        features = torch.cat([sum_rep, money.unsqueeze(-1), v.unsqueeze(-1)], dim=2)

        if self.normalize_obs:
            with torch.no_grad():
                self.rms_features.update(features)
                self.rms_condi.update(condi)
            features = self.rms_features.normalize(features)
            condi    = self.rms_condi.normalize(condi)

        return {"features": features, "condi": condi}

    @staticmethod
    def decode_actions(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        raw: (B,A,3) -> a∈(0,1), mu>0, h∈(0,1)  all shape (B,A)
        """
        a  = torch.sigmoid(raw[..., 0])
        mu = F.softplus(raw[..., 1]) + 1e-6
        h  = torch.sigmoid(raw[..., 2])
        return a, mu, h

    def compute_prices(self, savings: torch.Tensor, v: torch.Tensor, h: torch.Tensor):
        return self._compute_prices(savings, v, h)  # (wage, ret) as (B,A)

    def apply_taxes(self, wage, ret, v, h, a, delta=None, tax_params=None):
        # a 這裡代表「上期資本存量」或你 state 裡的 savings（非比例）
        money, ibt = self._calculate_money_disposable(
            wage=wage, ret=ret, v=v, h=h, k_prev=a, is_init=False
        )
        return money, ibt

    @staticmethod
    def output_transform(a: torch.Tensor, money_disposable: torch.Tensor):
        return EconVecEnv._output_transform(a, money_disposable)

    def transition_shocks(self, v_prev, is_superstar_prev, v_history, params=None, eps_v=None) -> tuple:
        """
        params 保留以後擴充；目前用 self.* 內部參數。
        """
        return self._transition_ability_batched(v_prev, is_superstar_prev, v_history, eps_v=eps_v)

    # ------------------- loss / reward（可選用） -------------------
    def loss_fn(self, state, actions, next_state) -> torch.Tensor:
        """
        r_t = sum_i [ CRRA(c_i) - disutil(h_i) ] ; shape (B,)
        連續制訓練時可以忽略，或用來做監督式 regularizer。
        """
        a, _, h = self.decode_actions(actions)
        c, _ = self.output_transform(a, next_state["moneydisposable"])

        eps = 1e-8
        if abs(self.theta - 1.0) < 1e-12:
            u_c = torch.log(c.clamp_min(eps))
        else:
            u_c = (c.clamp_min(eps) ** (1 - self.theta)) / (1 - self.theta)
        disutil = (h ** (1 + self.gamma)) / (1 + self.gamma)
        r = (u_c - disutil).sum(dim=1)  # (B,)
        return r

    # ------------------- stateful single-step（可用可不用） -------------------
    def step(self, actions: torch.Tensor):
        """
        一期前推並更新 internal state（*stateful*）。
        連續制任務：預設 done 永遠 False（除非設了 horizon>0）。
        Returns: obs_next, state_next, reward, done, info
        """
        assert actions.shape[-1] == 3, "actions last dim must be 3 (a, mu, h)"
        a, mu, h = self.decode_actions(actions)  # (B,A)

        wage, ret = self.compute_prices(self.state["savings"], self.state["v"], h)
        money, ibt = self.apply_taxes(wage, ret, self.state["v"], h, self.state["savings"])
        c, s = self.output_transform(a, money)
        v_next, super_next, _ = self.transition_shocks(self.state["v"], self.state["is_superstar"], v_history=None)

        next_state = {
            "moneydisposable": money,
            "savings": s,
            "v": v_next,
            "is_superstar": super_next,
            "a": a,
        }
        obs_next = self.build_obs(next_state)
        r = self.reward_fn(self.state, actions, next_state)

        self.state = next_state
        self.t += 1

        if self.horizon > 0 and self.t >= self.horizon:
            done = torch.ones(r.shape[0], dtype=torch.bool, device=self.device)
        else:
            done = torch.zeros(r.shape[0], dtype=torch.bool, device=self.device)

        info = self.metrics(self.state, next_state)
        info.update({
            "wage_mean": wage.mean().item(),
            "ret_mean": ret.mean().item(),
            "ibt_mean": ibt.mean().item(),
            "mu_mean":  mu.mean().item(),
            "t": self.t,
        })
        return obs_next, next_state, r, done, info

    # ------------------- stateless one-step（建議你訓練用） -------------------
    def step_from(self, state: dict, actions: torch.Tensor, eps_v: torch.Tensor | None = None,
                  v_history: torch.Tensor | None = None):
        """
        Stateless single-step transition from a *given* state.
        不會改 self.state；回傳 (next_state, outputs, v_history_next)。
        """
        a, mu, h = self.decode_actions(actions)
        wage, ret = self.compute_prices(state["savings"], state["v"], h)
        money, ibt = self.apply_taxes(wage, ret, state["v"], h, state["savings"])
        c, s = self.output_transform(a, money)
        v_next, super_next, v_hist_next = self.transition_shocks(
            state["v"], state["is_superstar"], v_history=v_history, eps_v=eps_v
        )
        next_state = {
            "moneydisposable": money,
            "savings": s,
            "v": v_next,
            "is_superstar": super_next,
            "a": a,
        }
        outputs = {"c": c, "wage": wage, "ret": ret, "ibt": ibt, "mu": mu}
        return next_state, outputs, v_hist_next

    def commit(self, next_state: dict, detach: bool = True):
        """
        把 next_state 設為 internal state。若 detach=True（建議訓練時使用），
        會切斷跨期計算圖，避免圖爆長。
        """
        self.state = {k: (v.detach() if detach else v) for k, v in next_state.items()}

    # ------------------- logging metrics -------------------
    def metrics(self, state, _next_state) -> dict:
        c, _ = self.output_transform(state.get("a", torch.zeros_like(state["savings"])),
                                     state["moneydisposable"])
        return {
            "consumption_mean": c.mean().item(),
            "savings_mean": state["savings"].mean().item(),
            "v_mean": state["v"].mean().item(),
            "superstar_share": state["is_superstar"].float().mean().item(),
        }