"""
Utility functions for saving and loading agents.

These functions help create standalone Python files for agent submission.
"""

import os
import numpy as np

def save_qlearning_agent(agent, output_path, agent_class_name="QLearningTrainedAgent"):
    """
    Save a trained Q-learning agent as a standalone Python file.
    
    Args:
        agent: The trained Q-learning agent instance
        output_path: Path where to save the agent file
        agent_class_name: Name for the agent class in the saved file
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Start building the file content
    file_content = f'''"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from agents.base_agent import BaseAgent

class {agent_class_name}(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # State discretization parameters
        self.position_bins = {agent.position_bins}
        self.wind_bins = {agent.wind_bins}
        
        # Q-table with learned values
        self.q_table = {{}}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
'''
    
    # Add all Q-values
    for state, values in agent.q_table.items():
        q_values_str = np.array2string(values, precision=4, separator=', ')
        file_content += f"        self.q_table[{state}] = np.array({q_values_str})\n"
    
    # Add remaining methods
    file_content += '''
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract position and wind from observation
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        
        # Discretize position (assume 32x32 grid)
        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # Discretize wind direction to 8 directions
        wind_direction = np.arctan2(wy, wx)  # Range: [-pi, pi]
        wind_bin = int(((wind_direction + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins
        
        # Return discrete state tuple
        return (x_bin, y_bin, wind_bin)
        
    def act(self, observation):
        """Choose the best action according to the learned Q-table."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Use default actions if state not in Q-table
        if state not in self.q_table:
            return 0  # Default to North if state not seen during training
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def reset(self):
        """Reset the agent for a new episode."""
        pass  # Nothing to reset
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
'''
    
    # Write the file
    with open(output_path, 'w') as f:
        f.write(file_content)
    
    print(f"Agent saved to {output_path}")
    print(f"The file contains {len(agent.q_table)} state-action pairs.")
    print(f"You can now use this file with validate_agent.ipynb and evaluate_agent.ipynb") 