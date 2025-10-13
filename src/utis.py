def mean_across_agents(x): # Since agent number is fix, thus we can use mean instead of sum
    return torch.mean(x, dim=1, keepdim=True)


def calculate_price(a, v, h):
    a_aggregate, l_aggregate_effective = mean_across_agents(a), mean_across_agents(h*v)
    wage = A * (1-alpha) * ((a_aggregate/l_aggregate_effective) ** alpha)
    ret = A * alpha * (a_aggregate/l_aggregate_effective ** alpha)
    return wage, ret

def taxfunc(ibt, abt, taxparams=TAX_PARAMS):
    it = ibt - (1 - taxparams["tax_income"]) * (ibt**(1-taxparams["income_tax_elasticity"])/(1-taxparams["income_tax_elasticity"])) # individual after tax income
    at = abt - (1-taxparams["tax_saving"]/1-taxparams["saving_tax_elasticity"]) * (abt**(1-taxparams["saving_tax_elasticity"])) # individual after tax saving
    return it, at

def calculate_moneydisposable(wage, ret, v, h, a, delta, is_init=False):
    if is_init:
        ibt = wage * h * v   # individual before tax income
    else:
        ibt = wage * h * v + (1-delta+ret) * a   # individual before tax income

    it, at = taxfunc(ibt = ibt, abt=a)
    money_disposable = it + at

    return money_disposable, ibt


def output_transform(a, money_disposable):

    # The a here is saving rate coming from the NN output
    consumption = money_disposable * (1 - a)
    savings = money_disposable * a
    return consumption, savings


def laborfocloss(a, h, ibt, money_disposable, wage, v, taxparams=TAX_PARAMS):

    loss_foc =  -h ** (-gamma) + ((1-a)*money_disposable/(1+taxparams["tax_consumption"])) * \
        (wage * v) * (1 - taxparams["tax_income"]) * (ibt ** (-taxparams["income_tax_elasticity"]))
    
    return torch.abs(loss_foc)

def transition_ability(
    v_prev: torch.Tensor,
    is_superstar_prev: torch.Tensor,
    v_history: torch.Tensor,
    rho_v: float,
    sigma_v: float,
    p: float,
    q: float,
    v_bar: float,
    v_min: float,
    v_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the ability value v_t for the next timestep based on the 
    Bewley-Aiyagari model with a normal and a super-star state.

    Args:
        v_prev (torch.Tensor): Ability values from the previous timestep (v_{t-1}).
        is_superstar_prev (torch.Tensor): Boolean tensor indicating which agents 
                                          were in the super-star state.
        v_history (torch.Tensor): A tensor containing the full history of 
                                  ability values for all agents.
        rho_v (float): Persistence parameter for the AR(1) process.
        sigma_v (float): Volatility parameter for the AR(1) process.
        p (float): Probability of transitioning from normal to super-star state.
        q (float): Probability of remaining in the super-star state.
        v_bar (float): Multiplier for super-star ability relative to the average.
        v_min (float): Minimum bound for the ability value.
        v_max (float): Maximum bound for the ability value.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the new ability 
                                           values (v_t) and the new superstar status.
    """
    num_households = v_prev.shape[0]
    
    # --- 1. Determine State Transitions (Logic Unchanged) ---
    
    transitions = torch.rand(num_households, device=v_prev.device)
    is_superstar_next = is_superstar_prev.clone()
    
    normal_to_superstar_mask = (~is_superstar_prev) & (transitions < p)
    is_superstar_next[normal_to_superstar_mask] = True
    
    superstar_to_normal_mask = is_superstar_prev & (transitions >= q)
    is_superstar_next[superstar_to_normal_mask] = False
    
    # --- 2. Calculate Next Ability v_t ---
    
    v_next = torch.zeros_like(v_prev)
    normal_mask_next = ~is_superstar_next
    superstar_mask_next = is_superstar_next
    
    # --- For agents in the NORMAL state next period (Logic Unchanged) ---
    if normal_mask_next.any():
        shocks = torch.randn(normal_mask_next.sum(), device=v_prev.device)
        log_v_next_normal = rho_v * torch.log(v_prev[normal_mask_next]) + sigma_v * shocks
        v_next_normal = torch.exp(log_v_next_normal)
        v_next[normal_mask_next] = torch.clamp(v_next_normal, min=v_min, max=v_max)

    # --- For agents in the SUPER-STAR state next period (Logic Unchanged) ---
    if superstar_mask_next.any():
        # Calculate the historical average from the provided history tensor.
        # Fallback to the previous period's average if history is empty (e.g., at the first step).
        if v_history is not None and v_history.numel() > 0:
            avg_ability = v_history.mean()
        else:
            avg_ability = v_prev.mean()

        v_next[superstar_mask_next] = v_bar * avg_ability
        
    return v_next, is_superstar_next

