# REINFORCEMENT LEARNING SAILING CHALLENGE

![Sailing Environment](illustration_challenge.png)

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/trahier/RL_project_sailing.git
cd RL_project_sailing
```

## Challenge Overview

Your mission is to develop an intelligent agent capable of navigating a sailboat from a starting point to a destination under varying wind conditions. This environment simulates sailing physics where the boat's movement is influenced by wind direction and intensity, requiring strategic planning to reach the goal efficiently.

**IMPORTANT**: This challenge requires you to **submit a pre-trained agent (policy) that maps observations to actions**, NOT a learning algorithm. Your submitted agent should be a fixed mapping from states (what the agent observes) to actions (a direction for the boat) that makes decisions based on the current observation without further learning during evaluation.

![Sailing Environment](sailing_environment.png)

The sailing environment features:
- A grid-based world where the boat must navigate from the starting point (bottom center) to the goal (top center)
- Realistic wind fields that vary spatially and temporally
- Physics-based boat movement influenced by wind direction and strength
- Success depends on understanding sailing physics and adapting to changing wind conditions

## Challenge description

You are provided with 3 (public) training wind scenarios with different spatial wind patterns. These three wind scenarios however have the same *temporal* behaviour (they are governed by the same Markov Decision Process). A final (hidden) wind scenario is kept secret, and will be used to evaluate your agent (it acts as a "test set" of a classical prediction challenge).

**This is the ultimate challenge of RL**: Training an agent on known environments and evaluating its ability to generalize to unseen conditions - the core goal of reinforcement learning.

## Installation

We recommend using a virtual environment:

```bash
# Create a virtual environment
python -m venv sailing-env

# Activate the virtual environment
# On Windows:
sailing-env\Scripts\activate
# On macOS/Linux:
source sailing-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Using Conda (Recommended for Anaconda/Miniconda users)

```bash
# Create a new conda environment
conda create -n sailing-env python=3.9

# Activate the environment
conda activate sailing-env

# Install dependencies
pip install -r requirements.txt
```

Note: We recommend Python 3.9 for optimal compatibility with all dependencies.

## Getting Started with the Challenge

We recommend exploring the notebooks in the following order:

1. **challenge_walkthrough.ipynb**  
   Introduction to the sailing challenge, environment mechanics, and agent interaction.

2. **design_agent.ipynb**  
   Learn how to develop your own sailing agent. This notebook guides you through creating either rule-based agents or training reinforcement learning agents.

3. **validate_agent.ipynb**  
   Test that your agent implementation meets the required interface.

4. **evaluate_agent.ipynb**  
   Evaluate your agent's performance across different wind scenarios.

5. **vizualise_agent.ipynb**
   This last notebook is useful to visualize the agents behaviour and therefore to have intuition on what might be the potential areas of improvement.

## Submission Instructions
Submissions must be made through the codabench platform. The private link to the challenge should have been shared to you separately. Should you have any question you can email: t.rahier at criteo.com

### Codabench Submission Requirements

To submit an agent to this challenge, upload a **ZIP file** whose root directory contains at least one `.py` file defining a class called `MyAgent`.

Your main file must start with this import and class declaration. IMPORTANT: the import is different from the agent you will have locally validated/evaluated/vizualised: locally we import `from agents.base_agent import BaseAgent` but the agent .py file to be submitted to codabench must import `from evaluator.base_agent import BaseAgent`.

```python
from evaluator.base_agent import BaseAgent

class MyAgent(BaseAgent):
    ...
```

This import is mandatory: it loads the official BaseAgent interface provided by the evaluation bundle. Your class must inherit from BaseAgent.

The key method to implement is:
```python
def act(self, obs, info=None):
    """Return an action given the current observation."""
    ...
```

### Validation and Evaluation

Before submitting, validate your agent:

```bash
cd src
python test_agent_validity.py path/to/your/agent.py
```

you can also evaluate it against one or several of the three training wind scenarios:

```bash
cd src
python3 evaluate_submission.py path/to/your/agent.py --wind_scenario training_1 --seeds 1 --num-seeds 100 --verbose
```

## Challenge Timeline

The timeline for your challenge is available on the codabench page.

## Communication

Please email me at t.rahier at criteo.com for any remarks / question. You can also use the codabench forum.
