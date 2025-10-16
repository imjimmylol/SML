# main.py
import argparse
import torch 
from src.configloader import load_config
from src.packenv import EconVecEnv
from src.env_pack_test import EconPackEnv
from src.utils import log_state_details, log_obs_details
from src.model import FiLMResNet2In

def main():
    parser = argparse.ArgumentParser(description="Run Bewley-MiLF Simulation")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    state_dim = 2*config.training.agents + 2
    cond_dim = 5

    model = FiLMResNet2In(
        state_dim=state_dim,
        cond_dim=cond_dim,
        hidden_dim=128,
        output_dim=3,
        num_res_blocks=2,
        dropout=0.1
    )

    env = EconPackEnv(
        agents=config.training.agents,
        tax_params=config.tax_params,
        v_min=config.shock.v_min,
        v_max=config.shock.v_max,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        seed=42,
    )
    
    state = env.reset(B=config.training.batch_size)
    obs = env.get_obs_all(state)

    obs_A, obs_B = obs['A'], obs['B']
    features_A, condi_A = obs_A.features, obs_A.condi
    features_B, condi_B = obs_B.features, obs_B.condi

    out_A, out_B = model(features_A, condi_A), model(features_B, condi_B)

    # --- 使用新的 utils 函式來記錄日誌 ---
    # log_state_details(state)
    # log_obs_details(obs)



if __name__ == "__main__":
    main()
