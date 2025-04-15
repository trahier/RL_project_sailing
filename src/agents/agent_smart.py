import numpy as np
from typing import Any, Dict
from agents.base_agent import BaseAgent
from sailing_physics import calculate_sailing_efficiency

class AgentSmart(BaseAgent):
    """
    A smarter agent that considers wind direction when choosing actions.
    The agent chooses the action that maximizes the dot product between:
    - The theoretical velocity (direction * sailing efficiency)
    - The vector pointing towards the goal
    """
    
    def __init__(self):
        """Initialize the agent."""
        super().__init__()
        self.name = "AgentSmart"
        self.goal_position = None
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()  # Call parent reset
        return None
    
    def act(self, observation: np.ndarray) -> int:
        """
        Choose an action based on the observation and wind direction.
        
        Args:
            observation: A numpy array containing [x, y, vx, vy, wx, wy]
                x, y: Current position
                vx, vy: Current velocity
                wx, wy: Wind vector at current position
        
        Returns:
            action: The action to take (0-7)
                0: North (up)
                1: Northeast
                2: East (right)
                3: Southeast
                4: South (down)
                5: Southwest
                6: West (left)
                7: Northwest
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
        return np.argmax(scores)
    
    def seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        super().seed(seed)
        return None
    
    def _calculate_best_action(self, position, wind_normalized, wind):
        """
        Calculate the best action by maximizing the dot product between theoretical velocity
        and the direction to the goal.
        
        Args:
            position: Current position
            wind_normalized: Normalized vector pointing TO where wind is coming FROM
            wind: The wind vector (unused, kept for interface consistency)
            
        Returns:
            best_action: The best action to take
        """
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
        
        # Vector to goal
        goal_vector = self.goal_position - position
        goal_distance = np.linalg.norm(goal_vector)
        if goal_distance > 0:
            goal_normalized = goal_vector / goal_distance
        else:
            goal_normalized = np.array([0, 1])  # Default to north if at goal
        
        # Calculate scores for each direction
        scores = []
        for i, direction in enumerate(directions):
            # Normalize direction
            direction_normalized = direction / np.linalg.norm(direction)
            
            # Calculate angle between wind source and direction
            wind_angle = np.arccos(np.clip(
                np.dot(wind_normalized, direction_normalized), -1.0, 1.0))
            
            # Calculate sailing efficiency based on angle to wind
            if wind_angle < np.pi/4:  # Less than 45 degrees to wind
                sailing_efficiency = 0.0  # Zero efficiency in no-go zone
            elif wind_angle < np.pi/2:  # Between 45 and 90 degrees
                sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)  # Linear increase to 1.0
            elif wind_angle < 3*np.pi/4:  # Between 90 and 135 degrees
                sailing_efficiency = 1.0  # Maximum efficiency
            else:  # More than 135 degrees
                sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4)  # Linear decrease
                sailing_efficiency = max(0.5, sailing_efficiency)  # But still decent
            
            # Calculate theoretical velocity (direction * efficiency)
            theoretical_velocity = direction_normalized * sailing_efficiency
            
            # Calculate score as dot product between theoretical velocity and goal direction
            score = np.dot(theoretical_velocity, goal_normalized)
            
            scores.append(score)
        
        # Choose the direction with the highest score
        best_action = np.argmax(scores)
        
        # If the best action has a very low score, consider staying in place
        if scores[best_action] < 0.1:
            return 8  # Stay in place
            
        return best_action
    
    def _direction_to_action(self, direction):
        """
        Convert a direction vector to an action.
        
        Args:
            direction: Direction vector [dx, dy]
            
        Returns:
            action: The corresponding action (0-7)
        """
        # Normalize for comparison
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        # Find the closest of the 8 directions
        dx, dy = direction
        
        # Map to closest of 8 directions
        if dx > 0.5 and dy > 0.5:
            return 1  # Northeast
        elif dx > 0.5 and abs(dy) <= 0.5:
            return 2  # East
        elif dx > 0.5 and dy < -0.5:
            return 3  # Southeast
        elif abs(dx) <= 0.5 and dy < -0.5:
            return 4  # South
        elif dx < -0.5 and dy < -0.5:
            return 5  # Southwest
        elif dx < -0.5 and abs(dy) <= 0.5:
            return 6  # West
        elif dx < -0.5 and dy > 0.5:
            return 7  # Northwest
        else:  # abs(dx) <= 0.5 and dy >= 0.5
            return 0  # North
    
    def _action_to_direction(self, action):
        """Convert action to direction vector."""
        if action == 8:  # Stay in place
            return np.array([0, 0])
        
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
        return directions[action] / np.sqrt(2) if action % 2 == 1 else directions[action]
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Path to save the agent's state
        """
        pass  # Nothing to save for this agent
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Path to load the agent's state from
        """
        pass  # Nothing to load for this agent 