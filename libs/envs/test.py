import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import time

# Import the custom environment
from jaxfluids_pino import Channel3DEnv  # Assuming the environment is saved as channel3d_env.py

def test_channel3d_env():
    # Initialize the environment
    env = Channel3DEnv(
        episode_length=1.0,
        action_length=0.1,
        reynolds_number=180,
        channel_type="minimal",
        observation_type="partial",
        render_mode="save",  # Save renderings to file
        is_double_precision=True
    )

    print("Environment initialized successfully.")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    try:
        # Reset the environment
        observation, info = env.reset(seed=42)
        print("\nEnvironment reset.")
        print(f"Initial observation shape: {observation['full'].shape if isinstance(observation, dict) else observation.shape}")
        print(f"Initial info: {info}")

        # Test a few steps with random actions
        print("\nTesting with random actions:")
        for step in range(3):
            # Sample a random action
            action = env.action_space.sample()
            print(f"\nStep {step + 1}:")
            print(f"Action shape: {action.shape}")

            # Perform a step
            start_time = time.time()
            observation, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - start_time

            # Print step results
            print(f"Observation shape: {observation['full'].shape if isinstance(observation, dict) else observation.shape}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}")
            print(f"Step time: {step_time:.2f} seconds")

            # Render the environment
            env.render()
            print("Rendered step visualization.")

            if terminated or truncated:
                print("Episode terminated or truncated early.")
                break

        # Test opposition control
        print("\nTesting with opposition control:")
        env.reset(seed=43)  # Reset for a clean state
        action = env.opposition_control(detection_plane_unit=0.1)
        print(f"Opposition control action shape: {action.shape}")

        # Perform a step with opposition control
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Observation shape: {observation['full'].shape if isinstance(observation, dict) else observation.shape}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        # Render the final state
        env.render()
        print("Rendered opposition control visualization.")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Clean up the environment
        env.close()
        print("\nEnvironment closed.")

if __name__ == "__main__":
    test_channel3d_env()