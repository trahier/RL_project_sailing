"""
Evaluation functions for sailing agents.

This module provides functions for evaluating and visualizing the performance
of sailing agents in the sailing environment.
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import Circle, Polygon # type: ignore
from IPython.display import display # type: ignore
import ipywidgets as widgets # type: ignore
from typing import List, Dict, Any, Tuple, Optional, Union, Callable # type: ignore
from env_sailing import SailingEnv
from agents.base_agent import BaseAgent
from IPython.display import clear_output # type: ignore
from tqdm.notebook import tqdm # type: ignore

def evaluate_agent(
    agent: BaseAgent,
    initial_windfield: Dict[str, Dict[str, Any]],
    seeds: Union[int, List[int]],
    max_horizon: int = 1000,
    verbose: bool = False,
    render: bool = False,
    full_trajectory: bool = False
) -> Dict[str, Any]:
    """Evaluate an agent on a specific initial windfield with given seeds.
    
    Args:
        agent: The agent to evaluate
        initial_windfield: Dictionary containing environment parameters
        seeds: Either a single seed or a list of seeds
        max_horizon: Maximum number of steps per episode
        verbose: Whether to show progress bar
        render: Whether to render frames
        full_trajectory: Whether to store all frames or just the last one
        
    Returns:
        Dictionary containing evaluation results
    """
    # Convert single seed to list
    single_seed = isinstance(seeds, int)
    if single_seed:
        seeds = [seeds]
    
    # Create environment using initial windfield parameters
    env_params = initial_windfield.get('env_params', {}).copy()  # Make a copy to avoid modifying the original
    
    # Set render_mode only if not already provided in env_params
    if 'render_mode' not in env_params and render:
        env_params['render_mode'] = "rgb_array"
    
    # Create the environment with all parameters
    env = SailingEnv(
        wind_init_params=initial_windfield['wind_init_params'],
        wind_evol_params=initial_windfield['wind_evol_params'],
        **env_params
    )
    
    # Initialize results
    all_rewards = []
    all_discounted_rewards = []  # Track discounted rewards
    all_steps = []
    all_successes = []
    frames = [] if render and single_seed else None
    positions = [] if single_seed else None
    actions = [] if single_seed else None
    individual_results = []  # Store detailed results for each episode
    
    # Evaluate on each seed
    iterator = seeds if not verbose else tqdm(seeds, desc="Evaluating seeds")
    for seed in iterator:
        # Reset environment and agent
        env.seed(seed)
        agent.seed(seed)
        observation, _ = env.reset()
        agent.reset()
        
        # Initialize episode variables
        total_reward = 0
        total_discounted_reward = 0  # Initialize discounted reward
        steps = 0
        episode_actions = []
        episode_success = False  # Default to failure
        positions_history = [] if single_seed else None
        
        # Run episode
        while steps < max_horizon:
            # Get action from agent
            action = agent.act(observation)
            episode_actions.append(action)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Calculate discounted reward using the environment's discount factor
            discounted_reward = reward * (env.reward_discount_factor ** steps)
            total_reward += reward
            total_discounted_reward += discounted_reward
            steps += 1
            
            # Store frame if rendering
            if render and single_seed:
                if full_trajectory:
                    frames.append(env.render())
                else:
                    # Only keep the last frame
                    frames = [env.render()]
            
            # Store position if single seed
            if single_seed:
                positions_history.append(info['position'].copy())
            
            # Check if episode is done
            if terminated or truncated:
                # Check if terminated due to success (reaching the goal) or timeout
                # A successful episode yields a reward of 100 in the current implementation
                episode_success = reward > 0 or (info.get('distance_to_goal', float('inf')) < 1.5)
                break
        
        # Store results
        all_rewards.append(total_reward)
        all_discounted_rewards.append(total_discounted_reward)
        all_steps.append(steps)
        all_successes.append(episode_success)
        if single_seed:
            positions = positions_history
            actions = episode_actions
        
        # Store individual episode results
        individual_results.append({
            'seed': seed,
            'reward': total_reward,
            'discounted_reward': total_discounted_reward,
            'steps': steps,
            'success': episode_success
        })
    
    # Compute statistics
    results = {
        'rewards': all_rewards,
        'discounted_rewards': all_discounted_rewards,
        'steps': all_steps,
        'success_rate': np.mean(all_successes),
        'mean_reward': np.mean(all_discounted_rewards),
        'std_reward': np.std(all_discounted_rewards),
        'mean_steps': np.mean(all_steps),
        'std_steps': np.std(all_steps),
        'individual_results': individual_results
    }
    
    # Add visualization data for single seed
    if single_seed:
        results['frames'] = frames if render else None
        results['positions'] = positions
        results['actions'] = actions
    
    return results

def visualize_trajectory(
    results: Dict[str, Any],
    env: SailingEnv,
    with_slider: bool = True
) -> None:
    """Visualize the trajectory of an agent.
    
    Args:
        results: Results dictionary from evaluate_agent
        env: The environment used for evaluation
        with_slider: Whether to show an interactive slider for frame selection
    """
    if not results.get('frames'):
        raise ValueError("No frames found in results. Did you run evaluate_agent with render=True?")
    
    frames = results['frames']
    
    if with_slider:
        from ipywidgets import interact, IntSlider # type: ignore
        
        def show_frame(frame_idx):
            plt.figure(figsize=(10, 10))
            plt.imshow(frames[frame_idx])
            plt.axis('off')
            plt.title(f'Step {frame_idx}')
            plt.show()
        
        interact(
            show_frame,
            frame_idx=IntSlider(
                min=0,
                max=len(frames)-1,
                step=1,
                value=0,
                description='Step:'
            )
        )
    else:
        # Show first and last frame side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(frames[0])
        ax1.set_title('Initial State')
        ax1.axis('off')
        ax2.imshow(frames[-1])
        ax2.set_title('Final State')
        ax2.axis('off')
        plt.show() 