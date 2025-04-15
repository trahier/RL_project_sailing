# Sailing Environment for Reinforcement Learning

This project implements a sailing navigation environment for reinforcement learning. It simulates a boat navigating from a starting point to a goal while accounting for wind dynamics.

## Project Structure

```
Sailing_project_v0/
├── notebooks/         # Jupyter notebooks for testing and demonstrations
├── src/               # Source code
│   ├── agents/        # Agent implementations
│   │   ├── agent_north.py    # Simple agent that always tries to move north
│   │   ├── agent_random.py   # Random agent for baseline comparison
│   │   └── agent_smart.py    # Smarter agent that considers wind direction
│   └── env_sailing.py        # The sailing environment implementation
└── README.md          # This file
```

## Environment Description

The sailing environment simulates a grid-based world where an agent (a boat) must navigate from a starting point (bottom center) to a goal (top center) while being affected by wind dynamics.

Key features:
- Grid-based world (configurable size, default 32x32)
- Wind dynamics with variable wind directions
- Physics-based boat movement influenced by wind
- Reward function based on progress toward the goal

## Sailing Physics

The environment simulates basic sailing physics with a focus on educational value. Here's how it works:

### Wind and Boat Interaction

Sailing relies on harnessing wind energy to propel a boat. The efficiency of this energy transfer depends primarily on the angle between the boat's direction and the wind direction:

- **Upwind (0-45°)**: In the "no-go zone," boats cannot sail directly into the wind. Efficiency is minimal (~10%).
- **Close-hauled (45-90°)**: Efficiency increases linearly from 50% to 100% as the angle approaches 90°.
- **Beam reach (90°)**: Maximum efficiency (100%) when wind is perpendicular to boat direction.
- **Broad reach (90-135°)**: Maintains maximum efficiency.
- **Running/Downwind (135-180°)**: Efficiency decreases but remains good (minimum 50%).

### Velocity Calculation

The boat's velocity at each timestep is calculated as:

```
new_velocity = (current_velocity × inertia_factor) + sailing_force
```

Where:
- **inertia_factor**: Determines how much of the current velocity is maintained (default: 0.8 or 80%)
- **sailing_force**: Calculated as `direction × efficiency × wind_strength × boat_efficiency`
  - **direction**: Normalized vector of desired movement direction
  - **efficiency**: Determined by the angle between boat direction and wind (as described above)
  - **wind_strength**: Magnitude of the wind vector at boat's position
  - **boat_efficiency**: How efficiently the boat converts wind power to movement (default: 0.2 or 20%)

The polar diagram in the visualization shows the resulting velocity achievable at different angles to the wind, considering these physics principles.

### Practical Implications

- To sail upwind, boats must "tack" (zig-zag) as they cannot sail directly into the wind
- Maximum speed is achieved when sailing perpendicular to the wind (beam reach)
- The boat's momentum (inertia) plays a significant role in maintaining smooth movement
- Higher boat efficiency values represent better boat designs that convert more wind energy into forward motion

## Agents

Three agents are provided:

1. **North Agent**: Always tries to move north (toward the goal). Simple but can be ineffective when sailing against the wind.
2. **Random Agent**: Chooses random actions, serving as a baseline for comparison.
3. **Smart Agent**: Considers wind direction when choosing actions, using sailing physics to navigate efficiently.

## Getting Started

To get started, explore the test notebook:

```
cd notebooks
jupyter notebook test_sailing_env.ipynb
```

## Example Usage

```python
import numpy as np
from src.env_sailing import SailingEnv
from src.agents.agent_smart import AgentSmart

# Create the environment
env = SailingEnv(grid_size=32, render_mode="rgb_array")

# Create the agent
agent = AgentSmart()

# Reset the environment and agent
observation, info = env.reset()
agent.reset()

# Run an episode
done = False
truncated = False
total_reward = 0

while not (done or truncated):
    # Choose an action
    action = agent.act(observation)
    
    # Take a step in the environment
    observation, reward, done, truncated, info = env.step(action)
    
    # Update total reward
    total_reward += reward
    
    # Render the environment
    env.render()

print(f"Episode finished with total reward: {total_reward}")
```

## Customization

You can customize the environment by adjusting parameters such as:
- `grid_size`: The size of the grid (default: 32)
- `wind_change_prob`: Probability of wind changing at each step (default: 0.1)
- `render_mode`: Rendering mode ("human", "rgb_array", or None)

## Creating New Agents

To create a new agent, implement a class with the following methods:
- `__init__()`: Initialize the agent
- `reset()`: Reset the agent's state
- `act(observation)`: Choose an action based on the observation
- `save(path)`: Save the agent's state
- `load(path)`: Load the agent's state

## License

This project is provided for educational purposes.