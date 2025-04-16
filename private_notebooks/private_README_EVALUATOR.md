# Sailing Challenge V1 - Evaluator Guide [PRIVATE]

This document provides instructions for evaluators of the Sailing Challenge V1. It includes information on the project structure, evaluation process, and how to use the provided tools.

## Project Overview

The Sailing Challenge is a reinforcement learning challenge where students implement an agent to navigate a sailboat from a starting point to a goal, taking into account wind conditions and sailing physics. The project includes:

1. A sailing environment (`SailingEnv`) with realistic physics
2. Multiple scenarios with different wind patterns
3. Example agents and evaluation tools
4. Comprehensive documentation and walkthroughs

## Directory Structure

```
Sailing_project_v1/
├── notebooks/
│   ├── demo_sailing.ipynb           - Comprehensive demo of the environment
│   ├── wind_evolution.ipynb         - Exploration of wind dynamics
│   ├── test_agent_interface.ipynb   - Testing different agent implementations
│   ├── evaluate_submission.ipynb    - Notebook for evaluating student submissions
│   ├── walkthrough_1_environment.ipynb - Student tutorial on environment
│   ├── walkthrough_2_agent_creation.ipynb - Guide for creating agents
│   └── walkthrough_3_evaluation.ipynb - Guide for evaluation
├── src/
│   ├── agents/
│   │   ├── base_agent.py            - Abstract base class for agents
│   │   ├── agent_north.py           - Simple agent that moves north
│   │   ├── agent_random.py          - Random action agent
│   │   ├── agent_smart.py           - Smart agent with physics knowledge
│   │   └── submission_example.py    - Template for student submissions
│   ├── env_sailing.py               - The sailing environment implementation
│   ├── evaluation.py                - Evaluation utilities
│   ├── sailing_physics.py           - Physics calculations for sailing
│   └── test_env.py                  - Environment tests
├── scenarios.py                     - Predefined wind scenarios
├── requirements.txt                 - Python dependencies
├── README.md                        - Public documentation for students
├── README_EVALUATOR.md              - Private documentation (this file)
└── WIND_EVOL_README.md              - Documentation about wind evolution
```

## Evaluation Process

### Submission Format

Students are required to submit:
1. A Python file containing their agent implementation (following the structure in `submission_example.py`)
2. Any additional files needed for their implementation (e.g., model weights)

The agent must inherit from `BaseAgent` and implement the required methods:
- `__init__()`: Initialize the agent
- `act(observation)`: Select an action based on the observation
- `reset()`: Reset the agent's internal state
- `seed(seed)`: Set the random seed for reproducibility
- `save(path)`: Save agent parameters (optional)
- `load(path)`: Load agent parameters (optional)

### Evaluation Steps

1. **Agent Validation**:
   - Check if the agent inherits from `BaseAgent`
   - Verify all required methods are implemented
   - Test if the agent returns valid actions

2. **Performance Evaluation**:
   - Evaluate the agent on the test scenario with multiple seeds
   - Calculate:
     - Success rate (% of episodes where the goal is reached)
     - Average reward
     - Average number of steps to reach the goal

3. **Scoring**:
   - Primary metric: Success rate on the test scenario
   - Secondary metrics: Average steps, average reward

### Using the Evaluation Tools

1. To evaluate a submission, use the `evaluate_submission.ipynb` notebook:
   - Update the `submission_path` variable to point to the student's file
   - Run the notebook to see comprehensive evaluation results

2. For automated evaluation of multiple submissions:
   - Use the functionality in `src/test_agent_validity.py`
   - Results will be saved in a structured format for easy comparison

## Common Issues and Solutions

1. **Invalid Action Format**: If an agent returns actions outside the range [0, 8], the environment will raise an error. Make sure students validate their action outputs.

2. **Missing Methods**: If any of the required methods are missing, the validation will fail with a specific error message.

3. **Import Errors**: If students use custom modules, ensure they are properly included with their submission.

4. **Poor Performance**: If an agent fails to reach the goal, provide feedback on potential improvements:
   - Consider wind direction and sailing physics
   - Implement more sophisticated exploration strategies
   - Use wind information more effectively

## Grading Guidelines

Suggested grading criteria:

1. **Functionality (40%)**:
   - Agent validation passes without errors
   - Agent returns valid actions consistently
   - Implementation follows the required interface

2. **Performance (40%)**:
   - Success rate on test scenario (primary metric)
   - Efficiency (average steps to goal)
   - Performance consistency across different seeds

3. **Code Quality (20%)**:
   - Code organization and readability
   - Proper documentation
   - Efficient implementation
   - Novel or creative approaches

## Advanced Topics

For advanced students, suggest exploring:
1. Implementing learning-based agents (RL, supervised learning)
2. Developing adaptive strategies for changing wind conditions
3. Optimizing for shortest paths while accounting for sailing physics
4. Creating multi-step planning algorithms

## Contact

For questions or issues with the challenge framework, please contact [Your Contact Information].

---

*This document is for evaluators only. Please do not share with students.* 