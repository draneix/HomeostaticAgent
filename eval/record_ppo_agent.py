# Create environment
from configs.config_ppo import PPOConfig
from envs.ant_env import HomeostaticAntEnv
from utils.utils_env import create_env
import torch
from utils.utils_ppo import HomeostaticPPO
import cv2
from skimage.transform import resize


def record_ppo():
    # Create environment
    config = PPOConfig(is_training=False, image_size=(512, 512))
    env = create_env(config, multiple_env=False)
    obs, info = env.reset()

    # Set up model
    print(f"Using device: {config.device}")
    agent = HomeostaticPPO(config).to(config.device)
    checkpoint = torch.load("models/PPO (30 epochs).pt", map_location=config.device)
    agent.load_state_dict(checkpoint)
    agent.eval()

    episode_frames_pov = []
    episode_frames_env = []

    episode_reward = []
    episode_food = []
    episode_water = []
    episode_steps = 0

    episode_hunger = []
    episode_thirst = []

    done = False
    while not done or episode_steps < config.max_episode_steps:
        # Prepare observation
        vision = resize(obs["vision"], (12, 64, 64), anti_aliasing=True, preserve_range=True)
        vision = torch.from_numpy(vision).unsqueeze(0).to(config.device)
        proprioception = torch.from_numpy(obs["proprioception"]).unsqueeze(0).to(config.device, dtype=torch.float32)
        internal_state = torch.from_numpy(obs["internal_state"]).unsqueeze(0).to(config.device)

        # Get action from agent
        with torch.no_grad():
            action, _, _, _ = agent(vision, proprioception, internal_state)
            action = action.cpu().numpy().squeeze(0)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward.append(reward)
        episode_food.append(int(info["food_consumed"]))
        episode_water.append(int(info["water_consumed"]))
        episode_hunger.append(info["hunger"])
        episode_thirst.append(info["thirst"])
        episode_steps += 1

        # Capture frames from info
        pov_frame = info["vision"]  # RGB frame
        # Convert to BGR for OpenCV and resize if needed
        pov_frame = cv2.cvtColor(pov_frame, cv2.COLOR_RGB2BGR)
        episode_frames_pov.append(pov_frame)

        env_frame = info["environment"]  # RGB frame with HUD
        # Convert to BGR for OpenCV
        env_frame = cv2.cvtColor(env_frame, cv2.COLOR_RGB2BGR)
        episode_frames_env.append(env_frame)

        done = terminated or truncated
        if episode_steps % 10 == 0:
            print(f"Step: {episode_steps}", end="\r", flush=True)
    print("Episode finished")
    env.close()

    # Save pov video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./videos/ppo_pov_video.mp4", fourcc, 30, (512, 512))
    for frame in episode_frames_pov:
        out.write(frame)
    out.release()

    # Save environment video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./videos/ppo_env_video.mp4", fourcc, 30, (512, 512))
    for frame in episode_frames_env:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    record_ppo()
