import torch
from utils.utils_env import create_env
from utils.utils_ppo import create_replay_buffer, add_to_replay_buffer, HomeostaticPPO
from configs.config_ppo import PPOConfig

def main():
    cfg = PPOConfig()
    env = create_env(cfg, async_env=True)
    replay_buffer = create_replay_buffer(cfg)
    agent = HomeostaticPPO(cfg)
    agent.to(cfg.device)
    
    obs, info = env.reset()
    for _ in range(6000):
        actions, log_prob, entropy, value = agent(obs["vision"], obs["proprioception"], obs["internal_state"])
        actions = actions.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()
        entropy = entropy.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations | truncations
        obs = next_obs
        add_to_replay_buffer(replay_buffer, obs, actions, rewards, done, value, log_prob)

    # Sample from replay buffer to validate    batch = replay_buffer.sample()
    print("Sampled batch from replay buffer:")
    sample_data = replay_buffer.sample()
    print(sample_data)

    print(f"Validation complete. Finished in {env.get_attr('current_step')}")
    env.close()


if __name__ == "__main__":
    main()
