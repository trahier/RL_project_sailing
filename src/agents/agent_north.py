import numpy as np
from typing import Any, Dict
from agents.base_agent import BaseAgent

class AgentNorth(BaseAgent):
    """
    A simple agent that always tries to move north (towards the top of the screen).
    This agent demonstrates how to navigate directly against the wind,
    which is typically difficult for sailing vessels.
    """
    
    def __init__(self):
        """Initialize the agent."""
        super().__init__()
        self.name = "AgentNorth"
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()  # Call parent reset
        pass  # Nothing else to reset for this simple agent
    
    def act(self, observation: np.ndarray) -> int:
        """
        Choose an action based on the observation.
        
        Args:
            observation: A numpy array containing the current observation.
                Format: [x, y, vx, vy, wx, wy] where:
                - (x, y) is the current position
                - (vx, vy) is the current velocity
                - (wx, wy) is the current wind vector
            
        Returns:
            action: The action to take (0 for North)
        """
        return 0  # Always take North action (up, increasing Y)
    
    def seed(self, seed: int = None) -> None:
        """Set the random seed for this agent.
        
        Args:
            seed: The random seed to use. If None, a random seed will be used.
        """
        super().seed(seed)  # Use parent's seed method
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Path to save the agent's state
        """
        pass  # Nothing to save for this simple agent
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Path to load the agent's state from
        """
        pass  # Nothing to load for this simple agent