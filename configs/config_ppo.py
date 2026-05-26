from dataclasses import dataclass, field
from typing import Tuple
import torch

from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class PPOConfig(EnvConfig):
    total_updates: int = 1_000
    rollout_steps: int = 2_000
    minibatch_size: int = 10_000
    batch_size: int = field(init=False)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 30
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_kl: float = 0.02

    def __post_init__(self):
        temp_batch_size = self.rollout_steps * self.num_workers
        object.__setattr__(self, 'batch_size', temp_batch_size)


# @dataclass
# class x:
#     total_updates: int = 1000
#     rollout_steps: int = 3000
#     batch_size: int = 30000
#     minibatch_size: int = 10000
#     gamma: float = 0.99
#     gae_lambda: float = 0.95
#     clip_coef: float = 0.2
#     ent_coef: float = 0.01
#     vf_coef: float = 0.5
#     max_grad_norm: float = 0.5
#     learning_rate_start: float = 1e-4
#     learning_rate_end: float = 1e-5
#     update_epochs: int = 4
#     target_kl: float = 0.02
#     eval_interval: int = 25
#     checkpoint_interval: int = 25
#     eval_episodes: int = 3
#     log_dir: str = "runs/homeostatic_ppo"
#     artifact_dir: str = "artifacts/homeostatic_ppo"
#     num_heat: int = 0
#     is_training: bool = True
#     reward_scale: float = 100.0
#     posture_drive_penalty: float = 0.0
#     movement_penalty_weight: float = 0.001
#     posture_penalty_weight: float = 0.005
#     max_steps: int = 60_000
#     image_size: Tuple[int, int] = (64, 64)
#     arena_size: float = 6.0
#     object_spacing: float = 2.0
#     object_interaction_dist: float = 1.0
#     heat_sensor_range: float = 1.0
#     num_food: int = 5
#     num_water: int = 5
#     day_night_cycle_len: int = 1
#     hunger_decay: float = 0.00015
#     thirst_decay: float = 0.00015
#     replenish_rate: float = 0.1
#     action_heat_gain_rate: float = 0.0
#     heat_source_gain_rate: float = 0.0
#     night_cooling_rate: float = 0.0
#     sweat_cooling_rate: float = 0.0
#     sweat_thirst_cost: float = 0.0
#     device: torch.device = torch.accelerator() if torch.has_accelerator() else torch.device("cpu")
