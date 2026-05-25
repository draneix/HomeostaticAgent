# Usage:
from utils.utils_env import create_env
from configs.config_env import EnvConfig
import numpy as np


def main():
    env = create_env(EnvConfig(), multiple_env=False)
    obs, info = env.reset()
    for _ in range(10):
        actions = env.action_space.sample()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        # print(f"Validation complete")
        # print(f"Observation space: {env.observation_space}")
        # print(f"Observation: {obs["proprioception"].max():.3f}, {obs["proprioception"].min():.3f}")
        print(f"Action space: {env.action_space}")
        print(f"Action: {actions}")
        # print(f"Action mean: {np.abs(actions).mean():.3f}")
        # print(f"Rewards: {rewards}")
        # print(f"Terminations: {terminations}")
        # print(f"Truncations: {truncations}")
        # print(f"Infos: {infos}")
    env.close()


if __name__ == "__main__":
    main()
