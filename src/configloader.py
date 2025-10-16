# src/config_loader.py
import yaml
import math
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert a dict into a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

def compute_shock_bounds(cfg):
    """Compute v_min and v_max from rho_v and sigma_v."""
    rho_v = cfg.shock.rho_v
    sigma_v = cfg.shock.sigma_v
    denom = math.sqrt(1 - rho_v ** 2)
    cfg.shock.v_min = math.exp(-2 * sigma_v / denom)
    cfg.shock.v_max = math.exp( 2 * sigma_v / denom)
    return cfg

def load_config(path: str = "config/default.yaml"):
    """Load YAML config, convert to object, and compute derived parameters."""
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg_dict)

    # --- Compute derived parameters ---
    cfg = compute_shock_bounds(cfg)

    return cfg
