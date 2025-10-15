# SAILING CHALLENGE

![Sailing Environment](illustration_challenge.png)

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/trahier/RL_project_sailing.git
cd RL_project_sailing
```

## Challenge Overview

Your mission is to develop an intelligent agent capable of navigating a sailboat from a starting point to a destination under varying wind conditions. This environment simulates sailing physics where the boat's movement is influenced by wind direction and intensity, requiring strategic planning to reach the goal efficiently.

**IMPORTANT**: This challenge requires you to **submit a pre-trained agent (policy) that maps observations to actions**, NOT a learning algorithm. Your submitted agent should be a fixed mapping from states to actions that makes decisions based on the current observation without further learning during evaluation.

![Sailing Environment](sailing_environment.png)

The sailing environment features:
- A grid-based world where the boat must navigate from the starting point (bottom center) to the goal (top center)
- Realistic wind fields that vary spatially and temporally
- Physics-based boat movement influenced by wind direction and strength
- Success depends on understanding sailing physics and adapting to changing wind conditions

## Challenge Progression

### Phase 1: Static Headwind Challenge (Until End of October)
**Current Focus!** Master the basics with a simple static headwind scenario.

The **Static Headwind** initial windfield provides a perfect environment for getting familiar with sailing:
- **Pure North wind** (headwind) with minimal variations
- **No wind evolution** - the wind pattern stays constant throughout the episode
- **Small spatial variations** in direction (slight NE/NW deviations) and amplitude
- **Perfect for discovering** basic sailing physics without complex wind dynamics

**Goal**: Create an agent that can successfully navigate from bottom to top against a static north wind using an appropriate strategy.

### Phase 2: Full Reinforcement Learning Challenge (After October)
Once you've mastered the static headwind, progress to the full RL challenge with dynamic wind conditions and multiple training scenarios.

The challenge provides 3 training initial windfields with different wind patterns. Your agent will be evaluated on both these training initial windfields and a hidden test initial windfield to assess its ability to generalize to new conditions.

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
   Evaluate your agent's performance across different initial windfields.

## Submission Instructions

### Primary Submission Method: Codabench

**Submit your agent via Codabench for automatic evaluation:**

<div align="center">
<h3>⭐ SUBMIT YOUR AGENT HERE ⭐</h3>
<h2><a href="https://www.codabench.org/competitions/11083/?secret_key=add817ed-6ae2-40f3-a31c-1e02f82d5201">Codabench Submission Link</a></h2>
</div>

**Important Note**: For now, the challenge is in its first phase (discovering the environment and agent design), and for this reason, agents are only evaluated against the static headwind on Codabench. In the next phase, more complex winds (such as the three training initial windfields and the hidden one) will be implemented on Codabench. Note that if challengers submit on the direct submission link OR on Codabench, I can also evaluate their agent on dynamic winds (including the hidden test wind) for curiosity.

#### Codabench Submission Requirements

To submit an agent to this challenge, upload a **ZIP file** whose root directory contains at least one `.py` file defining a class called `MyAgent`.

Your main file must start with this import and class declaration:
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

#### How to Submit on Codabench

1. **Prepare your agent**: Create a Python file with your `MyAgent` class
2. **Create a ZIP file**: Put your Python file in the root directory of a ZIP archive
3. **Upload to Codabench**: Use the submission link above
4. **Monitor results**: Check the leaderboard for your performance

### Alternative Submission Method: Direct Upload

If Codabench is not available, you can submit directly via Google Form:

<div align="center">
<h3>Alternative Submission Link</h3>
<h2><a href="https://forms.gle/nZCXLW5auGD56s8YA">Google Form Submission</a></h2>
</div>

**For direct submission, follow this naming convention:**
```
lastname_firstname_submission01.py
```

Use sequential numbering for multiple submissions (e.g., `lastname_firstname_submission02.py`).

### Validation and Evaluation

Before submitting, validate your agent:

```bash
cd src
python test_agent_validity.py path/to/your/agent.py
```

Then evaluate your agent on the static headwind:

```bash
cd src
python evaluate_submission.py path/to/your/agent.py --initial_windfield static_headwind --seeds 1 --num-seeds 100
```

## Challenge Timeline

**Phase 1 (Current - Until End of October)**:
- **Focus**: Static headwind challenge only
- **Evaluation**: Agents evaluated on static headwind scenario
- **Leaderboard Updates**: Regular updates during the upcoming two weeks

**Phase 2 (After October)**:
- **Focus**: Full reinforcement learning challenge
- **Training**: Multiple training initial windfields
- **Evaluation**: Training windfields + hidden test windfield
- **Duration**: Until Friday, December 12th
- **Leaderboard Updates**: Every Monday (showing performance on the test initial windfield)

## Communication

Please use the #rl-bootcamp-2025 chan to ask your questions and discuss any challenge-related topics.

## Good Luck!

Start by mastering the **Static Headwind** scenario to understand the basics of sailing physics and agent navigation. Once you can successfully navigate against a static headwind using an appropriate strategy, you'll be ready to tackle the full challenge with dynamic wind conditions!