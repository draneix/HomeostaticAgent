import torch
from utils.utils_env import create_env
from utils.utils_ppo import create_replay_buffer, add_to_replay_buffer
from configs.config_ppo import PPOConfig

def main():
    env = create_env(PPOConfig(), multiple_env=True)
    replay_buffer = create_replay_buffer(PPOConfig())
    obs, info = env.reset()
    for _ in range(6000):
        actions = env.action_space.sample()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations | truncations
        value = torch.tensor([0.0 for _ in range(env.num_envs)])      # Placeholder
        log_prob = torch.tensor([0.0 for _ in range(env.num_envs)])   # Placeholder
        add_to_replay_buffer(replay_buffer, obs, actions, rewards, done, value, log_prob)
        obs = next_obs

    # Sample from replay buffer to validate    batch = replay_buffer.sample()
    print("Sampled batch from replay buffer:")
    sample_data = replay_buffer.sample()
    print(sample_data)

    print(f"Validation complete. Finished in {env.get_attr('current_step')}")
    env.close()


if __name__ == "__main__":
    main()
