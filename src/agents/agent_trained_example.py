"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from agents.base_agent import BaseAgent

class QLearningTrainedAgent(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # State discretization parameters
        self.position_bins = 8
        self.wind_bins = 8
        
        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(4, 0, 0)] = np.array([8.3825e-04, 0.0000e+00, 0.0000e+00, 3.5882e-06, 0.0000e+00, 2.7285e-07,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(4, 1, 0)] = np.array([6.5747e-06, 0.0000e+00, 0.0000e+00, 2.6130e-06, 0.0000e+00, 0.0000e+00,
 3.7077e-03, 3.7501e-05, 0.0000e+00])
        self.q_table[(4, 2, 0)] = np.array([2.8863e-04, 0.0000e+00, 0.0000e+00, 5.0285e-06, 0.0000e+00, 0.0000e+00,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(4, 3, 0)] = np.array([0.0004, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(4, 4, 1)] = np.array([7.9966e-01, 0.0000e+00, 8.3296e-02, 0.0000e+00, 1.8734e-02, 1.5967e-05,
 9.8210e-02, 1.7600e-03, 0.0000e+00])
        self.q_table[(3, 4, 0)] = np.array([2.4129, 0.1175, 0.    , 0.    , 0.    , 0.1306, 0.    , 0.    , 0.    ])
        self.q_table[(3, 3, 0)] = np.array([0.6989, 0.    , 0.    , 0.    , 0.0323, 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 4, 1)] = np.array([3.3681, 0.8042, 0.3275, 0.2209, 1.077 , 0.6178, 0.1024, 1.7454, 0.3446])
        self.q_table[(3, 5, 1)] = np.array([11.5354,  2.6226,  1.522 ,  1.6556,  3.4948,  1.4659,  1.4421,  0.3304,
  4.0855])
        self.q_table[(2, 5, 0)] = np.array([0.0114, 0.    , 0.    , 0.    , 0.    , 0.    , 0.0004, 0.    , 0.    ])
        self.q_table[(2, 5, 1)] = np.array([0.   , 0.   , 0.   , 0.716, 0.   , 0.   , 0.   , 0.   , 0.   ])
        self.q_table[(2, 4, 0)] = np.array([0.0004, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 5, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 6, 0)] = np.array([0.0243, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(1, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 7, 1)] = np.array([0.2868, 0.0548, 0.1856, 0.0295, 0.109 , 0.0227, 0.0434, 0.0228, 0.0894])
        self.q_table[(0, 7, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 6, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 1)] = np.array([6.3013e-05, 3.1625e-06, 3.6102e-06, 8.8124e-07, 3.7064e-06, 1.3545e-06,
 2.2322e-06, 1.5854e-06, 2.6434e-06])
        self.q_table[(3, 0, 1)] = np.array([3.6739e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.2388e-07, 0.0000e+00,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(3, 1, 1)] = np.array([4.0618e-06, 1.9325e-06, 1.6066e-06, 7.0216e-06, 1.8679e-06, 0.0000e+00,
 5.2880e-04, 3.5380e-06, 3.0461e-06])
        self.q_table[(2, 1, 1)] = np.array([8.1049e-03, 0.0000e+00, 6.8781e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
 3.5792e-04, 0.0000e+00, 0.0000e+00])
        self.q_table[(2, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 3, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 3, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 2, 1)] = np.array([0.    , 0.    , 0.    , 0.0032, 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 2, 1)] = np.array([0.0201, 0.0018, 0.    , 0.0007, 0.0021, 0.0029, 0.0006, 0.    , 0.001 ])
        self.q_table[(1, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(2, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 2, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 1)] = np.array([2.4145e-06, 3.4068e-06, 4.1843e-08, 1.4448e-06, 2.9710e-06, 7.2152e-07,
 2.4481e-06, 7.0287e-06, 2.4840e-06])
        self.q_table[(3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 0)] = np.array([2.6803e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.2628e-05, 0.0000e+00,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(3, 2, 1)] = np.array([1.9696e-01, 0.0000e+00, 1.5235e-03, 0.0000e+00, 8.8255e-03, 4.2907e-06,
 8.4742e-03, 5.3817e-04, 5.4522e-04])
        self.q_table[(2, 2, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 2, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 3, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 4, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 3, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 2, 0)] = np.array([0.1414, 0.    , 0.    , 0.001 , 0.    , 0.0006, 0.    , 0.    , 0.    ])
        self.q_table[(2, 3, 1)] = np.array([0.0197, 0.1851, 0.0128, 0.0075, 0.0004, 0.0095, 0.    , 0.    , 0.    ])
        self.q_table[(2, 3, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 4, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 5, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 1)] = np.array([1.3406, 0.0108, 0.2517, 0.2039, 0.0105, 0.1227, 0.0157, 0.1839, 0.0192])
        self.q_table[(2, 4, 1)] = np.array([0.    , 0.    , 0.    , 0.4448, 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 5, 0)] = np.array([2.347 , 0.    , 0.    , 0.    , 0.    , 0.    , 0.1307, 0.0947, 0.0063])
        self.q_table[(2, 6, 0)] = np.array([0.2068, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0076])
        self.q_table[(2, 7, 1)] = np.array([1.0729, 0.2612, 0.4209, 0.1662, 0.1153, 0.3175, 0.1828, 0.185 , 0.8437])
        self.q_table[(1, 7, 0)] = np.array([0.0637, 0.3691, 0.0474, 0.0573, 0.032 , 0.0747, 0.0262, 0.0437, 0.0554])
        self.q_table[(5, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 1)] = np.array([4.2692e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
 0.0000e+00, 8.0618e-04, 0.0000e+00])
        self.q_table[(2, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 3, 1)] = np.array([0.0035, 0.0366, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(4, 5, 1)] = np.array([1.1322e+01, 9.6642e-03, 3.4079e-01, 3.9262e-01, 7.0656e-02, 1.0909e+00,
 3.6388e-01, 2.0323e+00, 3.1471e-01])
        self.q_table[(4, 4, 0)] = np.array([6.8004e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
 1.4171e-04, 0.0000e+00, 0.0000e+00])
        self.q_table[(5, 3, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 6, 1)] = np.array([0.0000e+00, 0.0000e+00, 3.2396e+01, 2.0938e-01, 1.2434e+00, 4.9286e-03,
 0.0000e+00, 0.0000e+00, 3.8379e-02])
        self.q_table[(1, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 3, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 1)] = np.array([46.2333, 12.4645, 15.8708,  7.4538, 12.7309, 12.2861,  9.9171, 18.2175,
  7.1803])
        self.q_table[(3, 7, 1)] = np.array([10.0176,  3.9371,  1.8794, 24.3754,  1.9326,  0.    ,  0.    ,  0.    ,
  1.0312])
        self.q_table[(3, 7, 0)] = np.array([0.3341, 0.2519, 0.3001, 0.2326, 0.2599, 0.2898, 0.3472, 0.3686, 0.2378])
        self.q_table[(5, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 6, 0)] = np.array([4.8270e-01, 1.1305e+01, 1.0046e-02, 2.1475e-01, 4.4425e-01, 0.0000e+00,
 1.0046e-02, 9.6770e-02, 1.3546e+00])
        self.q_table[(2, 7, 0)] = np.array([0.3229, 0.159 , 0.2775, 2.4718, 0.2379, 0.2523, 0.2015, 0.2   , 0.2032])
        self.q_table[(5, 5, 1)] = np.array([0.    , 0.    , 0.    , 0.    , 0.    , 0.3617, 0.    , 0.    , 0.    ])
        self.q_table[(4, 5, 0)] = np.array([2.1799, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(4, 6, 0)] = np.array([28.94  ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.4385])
        self.q_table[(4, 7, 0)] = np.array([91.7897,  0.    ,  0.    ,  0.    ,  1.2168, 11.218 , 25.0482, 11.2241,
  1.2464])
        self.q_table[(4, 7, 1)] = np.array([76.3369,  8.041 , 22.3372, 19.6321,  4.8453, 35.5163, 57.0308, 10.3095,
  7.3866])
        self.q_table[(5, 7, 1)] = np.array([9.0928, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(5, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 0)] = np.array([0.1945, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 9.4724, 0.    ])
        self.q_table[(5, 4, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 6, 0)] = np.array([0.098, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])
        self.q_table[(5, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

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
