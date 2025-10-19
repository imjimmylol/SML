# main.py
import argparse
import torch

from src.configloader import load_config
from src.env_pack_test import EconPackEnv
from src.env_buffer import EnvBuffer, _val
from src.utils import log_state_details
from src.model import FiLMResNet2In
from src.cpx_transition import create_shock_aware_transition

def main():
    parser = argparse.ArgumentParser(description="Run Bewley-MiLF Simulation for Two Worlds")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Create the configured transition function from our factory
    shock_transition_fn = create_shock_aware_transition(config)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"using device: {device}")
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
        v_history_max_len=config.training.training_steps
    )

    # This model is not used when generating dummy actions, but is kept for future use
    model = FiLMResNet2In(
        state_dim=2 * config.training.agents + 2,
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
    
        obs_A, obs_B = buffer.get_obs("A"), buffer.get_obs("B")
        actions_A, actions_B = model(obs_A.features, obs_A.condi), model(obs_B.features, obs_B.condi)

        # Generate dummy actions for demonstration
        B, A = config.training.batch_size, config.training.agents
        dummy_actions = {
            "consumption": torch.rand(B, A, device=device) * 0.1,
            "delta_savings": (torch.rand(B, A, device=device) - 0.5) * 0.2,
        }

        # In this setup, both worlds receive the same actions
        actions_A = dummy_actions
        actions_B = dummy_actions

        # Introduce the external shock on the first step to World B
        if t == 0:
            print("\n >> Applying initial shock to World B << ")
            v_B = _val(buffer.state_B, "v")
            buffer.update_state_value("B", "v", v_B * config.shock.v_bar)
            print("Value 'v' in World B has been shocked.")

        # Step the environment for both worlds using the new transition function
        info = buffer.step(actions_A, actions_B, shock_transition_fn)

        print("\nStep Info (World A):")
        for k, v in info["A"].items():
            print(f"  - {k}: shape={v.shape}, mean={v.mean():.4f}")

        print("\nStep Info (World B):")
        for k, v in info["B"].items():
            print(f"  - {k}: shape={v.shape}, mean={v.mean():.4f}")

        if t < 2: # Log first few steps to see the divergence
            print("\nUpdated State A:")
            log_state_details(buffer.state_A)
            print("\nUpdated State B:")
            log_state_details(buffer.state_B)

    print("\n---- Simulation Finished ----")
    print(f"Trajectory recorded for {len(buffer.traj_A)} steps in each world.")

if __name__ == "__main__":
    main()
