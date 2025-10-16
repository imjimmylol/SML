# main.py
import argparse
import torch
from src.configloader import load_config
from src.env_pack_test import EconPackEnv
from src.env_buffer import EnvBuffer, demo_transition # New import
from src.utils import log_state_details
from src.model import FiLMResNet2In

def main():
    parser = argparse.ArgumentParser(description="Run Bewley-MiLF Simulation with EnvBuffer")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # ---- 1. 初始化環境 ----
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
    
    # ---- 2. 使用 EnvBuffer 包裝 ----
    buffer = EnvBuffer(
        env=env,
        B=config.training.batch_size,
        keep_history=True,
    )

    # ---- 3. 初始化模型 ----
     # 2 state var broadcast to all agents + 2 individual state 
    state_dim = 2*config.training.agents + 2 
    model = FiLMResNet2In(
        state_dim=state_dim,
        cond_dim=len(config.tax_params), # tax conditions
        output_dim=3, # savings next period, multiplier, labor
        hidden_dim=128,
        dropout=0.1
    )

    # print("---- Initial State ----")
    # log_state_details(buffer.state)

    # ---- 4. 模擬迴圈 ----
    training_steps = config.training.training_steps
    for t in range(training_steps):
        print(f"""---- Step {t+1}/{training_steps} ----""")
        actions_A = model(buffer.get_obs(branch="A").features, buffer.get_obs(branch="A").condi)
        actions_B = model(buffer.get_obs(branch="B").features, buffer.get_obs(branch="B").condi)
        # print(buffer.get_obs_all()["A"])
        # print(buffer.get_obs_all()["B"])
        # print(buffer.get_obs_all().moneydisposable, buffer.get_obs_all().savings)
        # actions = model(buffer.get_obs_all().features, buffer.get_obs_all().condi)
        # print(actions)
        # print(f"Model Actions : { {k: v.shape for k, v in actions.items()} }")
        break
        # 在實際應用中，這裡會由模型產生 actions
        # 這裡我們用隨機值做為示意
        # B, A = config.training.batch_size, config.training.agents

        # dummy_actions = {
        #     "consumption": torch.rand(B, A, device=device) * 0.1,
        #     "delta_savings": (torch.rand(B, A, device=device) - 0.5) * 0.2,
        #     "growth_v1": torch.randn(B, A, device=device) * 0.01,
        #     "growth_v2": torch.randn(B, A, device=device) * 0.01,
        # }
        
        # 使用 buffer.step 和外部的 transition 函數來演化狀態
        info = buffer.step(dummy_actions, demo_transition)
        
        print("Step Info:")
        for k, v in info.items():
            print(f"  - {k}: shape={v.shape}, mean={v.mean():.4f}")
            
        print("\nUpdated State (sample):")
        log_state_details(buffer.state)

    print("\n---- Simulation Finished ----")
    print(f"Trajectory recorded for {len(buffer.traj)} steps.")
    if buffer.traj:
        print("First snapshot in trajectory:")
        for k, v in buffer.traj[0].items():
            print(f"  - {k}: shape={v.shape}")


if __name__ == "__main__":
    main()