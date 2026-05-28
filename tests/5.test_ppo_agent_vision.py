import os
from pathlib import Path

import cv2
import numpy as np
import torch

from configs.config_ppo import PPOConfig
from utils.utils_env import create_env
from utils.utils_ppo import HomeostaticPPO


def split_stacked_frames(stacked_vision: np.ndarray, stack_size: int) -> list[np.ndarray]:
    total_channels = stacked_vision.shape[0]
    if total_channels % stack_size != 0:
        raise ValueError(
            f"Stacked vision channels ({total_channels}) not divisible by stack_size ({stack_size})."
        )
    channels_per_frame = total_channels // stack_size
    return [
        stacked_vision[i * channels_per_frame : (i + 1) * channels_per_frame]
        for i in range(stack_size)
    ]


def confirm_stack_shift(prev_stacked: np.ndarray, curr_stacked: np.ndarray, stack_size: int) -> None:
    prev_frames = split_stacked_frames(prev_stacked, stack_size)
    curr_frames = split_stacked_frames(curr_stacked, stack_size)
    for idx in range(stack_size - 1):
        if not np.allclose(curr_frames[idx], prev_frames[idx + 1], atol=1e-6):
            raise AssertionError(
                f"Frame stack shift failed at index {idx}: "
                f"expected previous frame {idx + 1} to match current frame {idx}."
            )


def save_rgb_frames(stacked_vision: np.ndarray, stack_size: int, output_dir: Path, step: int) -> None:
    frames = split_stacked_frames(stacked_vision, stack_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        rgb = frame[:3]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        filename = output_dir / f"stack_step_{step}_frame_{idx}.png"
        cv2.imwrite(str(filename), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))


def main() -> None:
    cfg = PPOConfig(render_mode="rgb_array", is_training=True, frame_stack_size=3)
    env = create_env(cfg, multiple_env=False)

    agent = HomeostaticPPO(cfg)
    agent.to(cfg.device)

    obs, _ = env.reset()
    prev_vision = obs["vision"].copy()

    output_dir = Path(__file__).resolve().parents[1] / "checks" / "vision_stack"
    total_steps = 20

    for step in range(1, total_steps + 1):
        with torch.no_grad():
            vision = torch.from_numpy(obs["vision"]).float().unsqueeze(0).to(cfg.device)
            proprio = torch.from_numpy(obs["proprioception"]).float().unsqueeze(0).to(cfg.device)
            internal = torch.from_numpy(obs["internal_state"]).float().unsqueeze(0).to(cfg.device)
            actions, _, _, _ = agent(vision, proprio, internal)

        actions = actions.detach().cpu().numpy().squeeze(0)
        next_obs, _, terminated, truncated, _ = env.step(actions)

        confirm_stack_shift(prev_vision, next_obs["vision"], cfg.frame_stack_size)
        if step == 10:
            save_rgb_frames(next_obs["vision"], cfg.frame_stack_size, output_dir, step)

        prev_vision = next_obs["vision"].copy()
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            prev_vision = obs["vision"].copy()

    print("Frame stacking check passed. Captured frames saved to:", output_dir)
    env.close()


if __name__ == "__main__":
    main()
