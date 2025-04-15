"""A random agent that takes random actions."""

from agents.base_agent import BaseAgent
import numpy as np

class AgentRandom(BaseAgent):
    """Agent that takes random actions."""
    
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
    
    def act(self, observation: np.ndarray) -> int:
        """Select a random action."""
        return self.np_random.integers(0, 9)  # 0-8 inclusive
    
    def reset(self):
        """Reset the agent's state."""
        pass  # No state to reset for this agent
    
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