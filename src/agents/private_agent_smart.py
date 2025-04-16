"""
Example Submission for the Sailing Challenge

This file provides a template for students to create their own sailing agent.
Students should implement their solution by modifying this template.
"""

import numpy as np
from agents.base_agent import BaseAgent
from sailing_physics import calculate_sailing_efficiency

class SubmissionAgent(BaseAgent):
    """
    Example agent for the Sailing Challenge.
    
    This agent uses a physics-based strategy to navigate towards the goal,
    taking into account wind direction and sailing physics.
    Students should extend this template to implement more sophisticated strategies.
    """
    
    def __init__(self):
        """Initialize the agent with any required parameters."""
        super().__init__()
        self.np_random = np.random.default_rng()
        self.goal_position = None
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: A numpy array containing the current observation.
                Format: [x, y, vx, vy, wx, wy] where:
                - (x, y) is the current position
                - (vx, vy) is the current velocity
                - (wx, wy) is the current wind vector
        
        Returns:
            action: An integer in [0, 8] representing the action to take:
                - 0: Move North
                - 1: Move Northeast
                - 2: Move East
                - 3: Move Southeast
                - 4: Move South
                - 5: Move Southwest
                - 6: Move West
                - 7: Move Northwest
                - 8: Stay in place
        """
        # Extract relevant information from observation
        position = observation[0:2]  # x, y
        wind = observation[4:6]      # wx, wy
        
        # The goal is at the top center of the environment
        if self.goal_position is None:
            # Assuming a 32x32 grid, goal is at (16, 31)
            self.goal_position = np.array([16, 31])
        
        # Define all possible directions (corresponding to actions 0-7)
        directions = [
            np.array([0, 1]),     # 0: North (up)
            np.array([1, 1]),     # 1: Northeast
            np.array([1, 0]),     # 2: East (right)
            np.array([1, -1]),    # 3: Southeast
            np.array([0, -1]),    # 4: South (down)
            np.array([-1, -1]),   # 5: Southwest
            np.array([-1, 0]),    # 6: West (left)
            np.array([-1, 1]),    # 7: Northwest
        ]
        
        # Normalize wind vector
        wind_norm = np.linalg.norm(wind)
        if wind_norm > 0:
            wind_normalized = wind / wind_norm
        else:
            return 0  # If no wind, go north
            
        # Calculate vector to goal
        goal_vector = self.goal_position - position
        goal_distance = np.linalg.norm(goal_vector)
        if goal_distance > 0:
            goal_normalized = goal_vector / goal_distance
        else:
            return 0  # If at goal, go north
            
        # Calculate scores for each direction
        scores = []
        for i, direction in enumerate(directions):
            # Normalize direction
            direction_normalized = direction / np.linalg.norm(direction)
            
            # Calculate sailing efficiency
            efficiency = calculate_sailing_efficiency(direction_normalized, wind_normalized)
            
            # Calculate theoretical velocity
            theoretical_velocity = direction_normalized * efficiency
            
            # Score is dot product with goal direction
            score = np.dot(theoretical_velocity, goal_normalized)
            scores.append(score)
        
        # Return the action (0-7) with highest score
        best_action = np.argmax(scores)
        
        # Small chance of random action for exploration (optional)
        if self.np_random.random() < 0.05:  # 5% chance of random action
            return self.np_random.integers(0, 8)  # Random action between 0-7
            
        return best_action
    
    def reset(self) -> None:
        """Reset the agent's internal state between episodes."""
        self.goal_position = None
        
    def seed(self, seed: int = None) -> None:
        """Set the random seed for reproducibility."""
        self.np_random = np.random.default_rng(seed)
        
    def save(self, path: str) -> None:
        """
        Save the agent's learned parameters to a file.
        
        Args:
            path: Path to save the agent's state
        """
        # In a real implementation, students would save model parameters here
        # For example: np.save(path, self.model_parameters)
        pass
        
    def load(self, path: str) -> None:
        """
        Load the agent's learned parameters from a file.
        
        Args:
            path: Path to load the agent's state from
        """
        # In a real implementation, students would load model parameters here
        # For example: self.model_parameters = np.load(path)
        pass 