# Sailing Environment for Reinforcement Learning

This project implements a sailing navigation environment for reinforcement learning. It simulates a boat navigating from a starting point to a goal while accounting for wind dynamics.

## The Challenge

The Sailing Challenge tasks you with implementing an agent that can efficiently navigate a sailboat from a starting point to a goal under varying wind conditions. The agent needs to:

1. Understand and navigate based on sailing physics - where direction relative to wind determines sailing efficiency
2. Adapt to changing wind patterns that vary across different scenarios
3. Find efficient paths considering that the shortest path may not be the most effective due to wind constraints
4. Develop a strategy that generalizes well across multiple wind scenarios

![Sailing Environment](sailing_environment.png)

## Installation

To set up the environment, we recommend using a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Sailing_project_v1/
├── notebooks/         # Jupyter notebooks for testing and demonstrations
├── src/               # Source code
│   ├── agents/        # Agent implementations
│   │   ├── agent_greedy.py   # Simple greedy agent example
│   │   └── base_agent.py     # Base class for all agents
│   ├── scenarios/     # Wind scenario definitions
│   ├── env_sailing.py        # The sailing environment implementation
│   ├── evaluation.py         # Tools for evaluating agents
│   └── sailing_physics.py    # Physics calculations for sailing
├── requirements.txt   # Required Python packages
└── README.md          # This file
```

## Grading

Your submission will be evaluated based on:

- **Primary (70%)**: Mean reward achieved on the hidden test scenario with novel wind patterns
- **Secondary (30%)**: Performance across the training scenarios
- **Bonus**: Code quality, novel approaches, and efficiency (steps to goal)

A higher mean reward indicates a more effective sailing strategy. The best agents will achieve high performance on both known (training) and unknown (test) scenarios, demonstrating their ability to generalize.

## Getting Started: Notebook Reading Order

We recommend exploring the notebooks in this order:

1. **challenge_walkthrough.ipynb**: An introduction to the sailing challenge, environment mechanics, and how agents interact with the environment
   
2. **wind_evolution_exploration.ipynb**: Deep dive into how wind patterns evolve and their impact on navigation
   
3. **validate_agent.ipynb**: Learn how to implement a valid agent and confirm it meets the interface requirements
   
4. **evaluate_agent.ipynb**: Understand how to evaluate your agent's performance across different scenarios

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

- **Upwind (0-45°)**: In the "no-go zone," boats cannot sail directly into the wind. Efficiency is minimal (~5%).
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
- **inertia_factor**: Determines how much of the current velocity is maintained
- **sailing_force**: Calculated as `direction × efficiency × wind_strength × boat_efficiency`

## Agents

The project includes a simple example agent:

**Greedy Agent**: A basic agent in `agent_greedy.py` that always tries to move north (toward the goal). Simple but can be ineffective when sailing against the wind. This serves as a starting point for developing your own agent.

## Developing Better Agents

To develop an agent that outperforms the greedy baseline:

1. **Start with a wind-aware agent**: Implement a simple agent that considers the current wind direction and chooses actions based on sailing efficiency.

2. **Add path planning**: Implement algorithms like A* that account for both distance to goal and sailing efficiency.

3. **Consider future wind states**: If your agent can predict or account for wind changes, it can make more informed decisions.

4. **Apply reinforcement learning**: For more advanced solutions, consider:
   - Q-learning or SARSA for discrete state-action spaces
   - Deep Q-Networks for handling the continuous wind field
   - Policy gradient methods to directly learn optimal sailing policies

5. **Test thoroughly**: Evaluate your agent across all training scenarios to ensure it generalizes well.

## Example Usage

```python
import numpy as np
from src.env_sailing import SailingEnv
from src.agents.agent_greedy import GreedyAgent

# Create the environment
env = SailingEnv(grid_size=32, render_mode="rgb_array")

# Create the agent
agent = GreedyAgent()

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

## Command Line Interface

The project includes command-line tools for agent validation and evaluation:

### Validating an Agent

To validate that your agent implementation meets the interface requirements:

```bash
cd Sailing_project_v1/src
python test_agent_validity.py path/to/your_agent.py
```

Example output for a valid agent:
```
Agent validation results for: path/to/your_agent.py
Agent name: YourAgent
Valid: True
Validation successful! The agent meets all requirements.
```

### Evaluating an Agent

To evaluate your agent's performance on different scenarios:

```bash
cd Sailing_project_v1/src
python evaluate_submission.py path/to/your_agent.py --scenario training_1 --seeds 42 43 44
```

Options:
- `--scenario`: Scenario to evaluate (default: simple_test)
- `--seeds`: Seeds to use for evaluation (default: 42)
- `--max_horizon`: Maximum steps per episode (default: 200)
- `--output`: Save results to specified JSON file
- `--verbose`: Show detailed progress

## Creating New Agents

To create a new agent, implement a class that inherits from `BaseAgent` with the following methods:
- `__init__()`: Initialize the agent
- `reset()`: Reset the agent's state
- `act(observation)`: Choose an action based on the observation
- `seed(seed)`: Set the random seed for reproducibility

## License

This project is provided for educational purposes.