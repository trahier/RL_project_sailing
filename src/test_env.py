import numpy as np
from env_sailing import SailingEnv

def test_env():
    """Test if the SailingEnv class works correctly."""
    print("Creating environment...")
    env = SailingEnv(grid_size=32, render_mode="rgb_array")
    
    print("Resetting environment...")
    observation, info = env.reset()
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    print("Taking steps...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}, Action: {action}, Reward: {reward}")
        
        # Check observation
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key} shape: {value.shape}")
            else:
                print(f"  {key}: {value}")
    
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_env() 