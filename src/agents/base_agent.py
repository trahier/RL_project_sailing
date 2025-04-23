"""Base agent class that all agents must inherit from."""

from abc import ABC, abstractmethod
import numpy as np # type: ignore
from typing import Dict, Any, Tuple, Optional

class BaseAgent(ABC):
    """Abstract base class for all sailing agents.
    
    This class defines the interface that all agents must implement.
    Agents are expected to make decisions based on the current observation
    of the environment state.
    """
    
    def __init__(self):
        """Initialize the agent."""
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """Select an action based on the current observation.
        
        Args:
            observation: A numpy array containing the current observation.
                Format: [x, y, vx, vy, wx, wy] where:
                - (x, y) is the current position
                - (vx, vy) is the current velocity
                - (wx, wy) is the current wind vector
        
        Returns:
            action: An integer in [0, 8] representing the action to take:
                - 0-7: Move in that direction (0=N, 1=NE, 2=E, etc.)
                - 8: Stay in place
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset the agent's internal state.
        
        This method is called at the beginning of each episode.
        Override this method if your agent maintains internal state.
        """
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed for this agent.
        
        Args:
            seed: The random seed to use. If None, a random seed will be used.
        """
        self.np_random = np.random.default_rng(seed) 