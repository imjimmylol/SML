# main.py
import argparse
import torch
from src.configloader import load_config
from src.env_pack_test import EconPackEnv
from src.env_buffer import EnvBuffer, demo_transition, _val
from src.utils import log_state_details
from src.model import FiLMResNet2In

def main():
    parser = argparse.ArgumentParser(description="Run Bewley-MiLF Simulation for Two Worlds")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = EconPackEnv(
        agents=config.training.agents,
        tax_params=config.tax_params,
        v_min=config.shock.v_min,
        v_max=config.shock.v_max,
        device=device,
        dtype=torch.float32,
        seed=42,
    )
    
    buffer = EnvBuffer(
        env=env,
        B=config.training.batch_size,
        keep_history=True,
    )

    state_dim = 2 * config.training.agents + 2
    model = FiLMResNet2In(
        state_dim=state_dim,
        cond_dim=len(config.tax_params),
        output_dim=3,
        hidden_dim=128,
        dropout=0.1
    ).to(device)

    print("---- Initial States (A and B are identical) ----")
    print("State A:")
    log_state_details(buffer.state_A)

    training_steps = config.training.training_steps
    for t in range(training_steps):
        print(f"\n---- Step {t+1}/{training_steps} ----")

        # 1. Get observations for both worlds (optional, not used for dummy actions)
        # obs_A = buffer.get_obs(branch="A")
        # obs_B = buffer.get_obs(branch="B")

        # 2. Generate dummy actions for demonstration
        print("\n>> Using randomized dummy actions for this step. <<")
        B, A = config.training.batch_size, config.training.agents
        dummy_actions_A = {
            "consumption": torch.rand(B, A, device=device) * 0.1,
            "delta_savings": (torch.rand(B, A, device=device) - 0.5) * 0.2,
            "growth_v": torch.randn(B, A, device=device) * 0.01,
        }
        dummy_actions_B = {
            "consumption": torch.rand(B, A, device=device) * 0.1,
            "delta_savings": (torch.rand(B, A, device=device) - 0.5) * 0.2,
            "growth_v": torch.randn(B, A, device=device) * 0.01,
        }

        # 3. Introduce the shock after the first step
        if t == 0:
            print("\n >> Applying initial shock to World B << ")
            # Example shock: increase v in world B by 10%
            shock_multiplier = 1.1 
            v_B = _val(buffer.state_B, "v")
            buffer.update_state_value("B", "v", v_B * shock_multiplier)
            print("Value 'v' in World B has been shocked.")

        # 4. Step the environment for both worlds using the demo transition
        info = buffer.step(dummy_actions_A, dummy_actions_B, demo_transition)

        print("\nStep Info (World A):")
        for k, v in info["A"].items():
            print(f"  - {k}: shape={v.shape}, mean={v.mean():.4f}")

        print("\nStep Info (World B):")
        for k, v in info["B"].items():
            print(f"  - {k}: shape={v.shape}, mean={v.mean():.4f}")

        # Optional: Log state details to see divergence
        if t < 2: # Log first few steps to see the divergence
            print("\nUpdated State A:")
            log_state_details(buffer.state_A)
            print("\nUpdated State B:")
            log_state_details(buffer.state_B)

    print("\n---- Simulation Finished ----")
    print(f"Trajectory recorded for {len(buffer.traj_A)} steps in each world.")
