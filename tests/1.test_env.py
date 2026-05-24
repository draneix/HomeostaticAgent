# Usage:
from utils.utils_env import create_env
from configs.config_env import EnvConfig


def main():
    env = create_env(EnvConfig(), multiple_env=True)
    obs, info = env.reset()
    for _ in range(6000):
        actions = env.action_space.sample()
        obs, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Validation complete. Finished in {env.get_attr('current_step')}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Action: {actions}")
    print(f"Rewards: {rewards}")
    print(f"Terminations: {terminations}")
    print(f"Truncations: {truncations}")
    print(f"Infos: {infos}")
    env.close()


if __name__ == "__main__":
    main()
