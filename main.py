# main.py
import argparse
import torch 
from src.configloader import load_config
from src.packenv import EconVecEnv

def main():
    parser = argparse.ArgumentParser(description="Run Bewley-MiLF Simulation")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    env = EconVecEnv(agents=config.training.agents, 
                     config=config, 
                     normalize_obs=False # 這個參數之後要改掉，這邊是先可run 再標準化
                     )

    # 初始化（只需要一次 reset）
    B = config.training.batch_size
    obs0, state, _ = env.reset(batch_size=B)

    # print(obs0.keys())
    # print(state.keys())

    for i in range(config.training.training_steps):
        # 兩條互不相關的 ability shocks
        eps_v_a = torch.randn(B, env.A, device=env.device)
        eps_v_b = torch.randn(B, env.A, device=env.device)

        # 若你有 policy，可先用 obs → actions；這裡示意隨機 actions
        actions = torch.randn(B, env.A, 3, device=env.device)

        # 從同一起點 state 出發，分別前推 A/B
        state_a, out_a, env.v_history_A = env.step_from(state, actions, eps_v=eps_v_a, v_history=env.v_history_A)
        state_b, out_b, env.v_history_B = env.step_from(state, actions, eps_v=eps_v_b, v_history=env.v_history_B)

        # loss, Residual1, Residual2 = loss_fn(out_a, out_b)  # 你的自定義損失
        # if i % TRAIN_STEP_INTERVAL == 0:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # 以 A 路徑作為真實演化（避免跨期回傳就 detach）
        env.commit(state_a, detach=True)
        state = env.state

        # if i % config.training.display_step == 0:
        #     print(f"Loss {i}: {loss.item()} (R1={Residual1.item()}, R2={Residual2.item()})")

if __name__ == "__main__":
    main()
