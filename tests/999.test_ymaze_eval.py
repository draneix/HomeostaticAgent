import cv2
from configs.config_ymaze import YMazeConfig
from envs.ymaze_test_env import YMazeTestEnv


def test_eval_env():
    cfg = YMazeConfig(render_mode="human", image_size=(512, 512), is_training=False)
    env = YMazeTestEnv(cfg)
    obs = env.reset()

    try:
        while True:
            action = env.action_space.sample()  # Random action for testing
            obs, reward, terminated, truncated, info = env.step(action)
            print(info.keys())
            # Extract the POV image - RGB part only for visualization
            pov_image = info["vision"]
            # Convert RGB to BGR for OpenCV display
            pov_bgr = cv2.cvtColor(pov_image, cv2.COLOR_RGB2BGR)
            # Display the POV
            cv2.imshow("Ant POV Perspective", pov_bgr)
            # Also show the side-view environment camera in another window
            env_bgr = cv2.cvtColor(info["environment"], cv2.COLOR_RGB2BGR)
            cv2.imshow("Global Environment View", env_bgr)
            # Wait for 1ms and check if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs = env.reset()
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        print("Environment closed and windows destroyed.")


if __name__ == "__main__":
    test_eval_env()
