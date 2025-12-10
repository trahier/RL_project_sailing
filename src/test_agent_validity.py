"""
Test Agent Validity

This module provides functions to test if a sailing agent meets the required
interface specifications and behaves correctly.

Usage:
    python src/test_agent_validity.py path/to/your_agent.py
    python src/test_agent_validity.py path/to/your_agent.py --verbose

Examples:
    python src/test_agent_validity.py src/agents/agent_naive.py
    python src/test_agent_validity.py my_custom_agent.py --verbose
"""

import os
import sys
import importlib.util
import inspect
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Type

# Define color codes for terminal output
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('..'))

# Import with proper path handling
try:
    from agents.base_agent import BaseAgent
    from env_sailing import SailingEnv
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agents.base_agent import BaseAgent
        from env_sailing import SailingEnv
    except ImportError:
        print(f"{RED}Error: Unable to import required modules.{RESET}")
        print("Make sure you're running this script from the repository root or src directory.")
        sys.exit(1)

# Try different approaches to import wind_scenarios
try:
    # Direct import from the package
    from wind_scenarios import get_wind_scenario
except ImportError:
    # Relative import in case the script is run from the src directory
    try:
        from wind_scenarios import get_wind_scenario
    except ImportError:
        print(f"{RED}Error: Unable to import wind_scenarios module.{RESET}")
        print("Make sure you're running this script from the repository root or src directory.")
        sys.exit(1)


class AgentValidityError(Exception):
    """Exception raised when an agent does not meet the validity requirements."""
    pass


def load_agent_class(filepath: str) -> Type[BaseAgent]:
    """
    Load an agent class from a Python file.
    
    Args:
        filepath: Path to the Python file containing the agent class
        
    Returns:
        The agent class (not an instance)
        
    Raises:
        AgentValidityError: If the file does not contain a valid agent class
    """
    # Get the file name without extension to use as module name
    module_name = Path(filepath).stem
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise AgentValidityError(f"Could not load module from {filepath}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find agent classes in the module (classes that inherit from BaseAgent)
        agent_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseAgent) and 
                obj != BaseAgent):
                agent_classes.append(obj)
        
        if not agent_classes:
            raise AgentValidityError(
                f"No agent class found in {filepath}. "
                "Make sure your agent class inherits from BaseAgent."
            )
        
        if len(agent_classes) > 1:
            print(f"{YELLOW}Warning: Multiple agent classes found in {filepath}. "
                  f"Using {agent_classes[0].__name__}{RESET}")
        
        return agent_classes[0]
    
    except (ImportError, SyntaxError) as e:
        raise AgentValidityError(f"Error importing agent file: {str(e)}")


def check_required_methods(agent_class: Type[BaseAgent]) -> List[str]:
    """
    Check if an agent class implements all required methods.
    
    Args:
        agent_class: The agent class to check
        
    Returns:
        A list of missing or incorrect methods, empty if all required methods are present
    """
    missing_methods = []
    
    # Required methods and their expected signatures
    required_methods = {
        'act': ['observation'],
        'reset': [],
        'seed': ['seed'],
    }
    
    for method_name, expected_params in required_methods.items():
        # Check if method exists
        if not hasattr(agent_class, method_name):
            missing_methods.append(f"Missing required method: {method_name}")
            continue
        
        # Get method
        method = getattr(agent_class, method_name)
        
        # Check if it's a method (not an attribute)
        if not callable(method):
            missing_methods.append(f"{method_name} is not callable")
            continue
        
        # Check method signature
        try:
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            # Remove 'self' from params if present
            if params and params[0] == 'self':
                params = params[1:]
            
            # Check if all required parameters are present
            for param in expected_params:
                if param not in params:
                    missing_methods.append(
                        f"Method {method_name} is missing required parameter: {param}"
                    )
        except (ValueError, TypeError):
            missing_methods.append(f"Could not inspect method: {method_name}")
    
    return missing_methods


def test_agent_actions(agent_instance: BaseAgent) -> List[str]:
    """
    Test if an agent instance returns valid actions for sample observations.
    
    Args:
        agent_instance: The agent instance to test
        
    Returns:
        A list of action validity issues, empty if all actions are valid
    """
    issues = []
    
    # Create a test environment
    env = SailingEnv()
    observation, _ = env.reset(seed=42)
    
    # Test with multiple observations
    for _ in range(10):
        try:
            # Get action from agent
            action = agent_instance.act(observation)
            
            # Check if action is an integer
            if not isinstance(action, (int, np.integer)):
                issues.append(
                    f"Agent returned non-integer action: {action} (type: {type(action)}). "
                    f"The act() method must return an integer between 0 and 8."
                )
                break
            
            # Check if action is in valid range
            if action < 0 or action > 8:
                issues.append(
                    f"Agent returned out-of-range action: {action}. "
                    f"Valid actions are integers from 0 to 8."
                )
                break
            
            # Take a step in the environment
            observation, _, terminated, truncated, _ = env.step(action)
            
            # Break if episode is done
            if terminated or truncated:
                break
                
        except Exception as e:
            issues.append(f"Error during agent action: {str(e)}")
            break
    
    return issues


def validate_agent(filepath: str) -> Dict[str, Any]:
    """
    Validate an agent implementation.
    
    Args:
        filepath: Path to the Python file containing the agent
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'agent_class': class or None,
            'agent_name': str or None,
            'errors': list of error messages,
            'warnings': list of warning messages
        }
    """
    results = {
        'valid': False,
        'agent_class': None,
        'agent_name': None,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Step 1: Load the agent class
        agent_class = load_agent_class(filepath)
        results['agent_class'] = agent_class
        results['agent_name'] = agent_class.__name__
        
        # Step 2: Check required methods
        missing_methods = check_required_methods(agent_class)
        if missing_methods:
            results['errors'].extend(missing_methods)
            return results
        
        # Step 3: Instantiate the agent
        try:
            agent_instance = agent_class()
        except Exception as e:
            results['errors'].append(
                f"Failed to instantiate agent: {str(e)}\n"
                f"Make sure your agent's __init__ method accepts no arguments."
            )
            return results
        
        # Step 4: Test agent actions
        action_issues = test_agent_actions(agent_instance)
        if action_issues:
            results['errors'].extend(action_issues)
            return results
        
        # All tests passed
        results['valid'] = True
        
        # Optional: Check for save/load methods
        if not hasattr(agent_class, 'save') or not callable(getattr(agent_class, 'save')):
            results['warnings'].append("Agent does not implement save() method")
        
        if not hasattr(agent_class, 'load') or not callable(getattr(agent_class, 'load')):
            results['warnings'].append("Agent does not implement load() method")
            
    except AgentValidityError as e:
        results['errors'].append(str(e))
    except Exception as e:
        results['errors'].append(f"Unexpected error: {str(e)}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a sailing agent implementation against requirements."
    )
    
    parser.add_argument(
        "agent_file",
        nargs="?",
        help="Path to the Python file containing your agent implementation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show more detailed validation information"
    )
    
    return parser.parse_args()


def print_validation_results(results, verbose=False, filepath=None):
    """Print validation results in a user-friendly format."""
    print(f"\n🤖 {BOLD}Agent: {results['agent_name'] or 'Unknown'}{RESET}")
    
    if results['valid']:
        print(f"\n{GREEN}✅ SUCCESS: Your agent meets all requirements!{RESET}")
    else:
        print(f"\n{RED}❌ FAILURE: Your agent does not meet all requirements.{RESET}")
    
    if results['errors']:
        print(f"\n{RED}🛑 Errors that must be fixed:{RESET}")
        for i, error in enumerate(results['errors'], 1):
            print(f"  {i}. {error}")
    
    if results['warnings']:
        print(f"\n{YELLOW}⚠️ Warnings (not required to fix, but recommended):{RESET}")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"  {i}. {warning}")
    
    if verbose and results['valid']:
        print(f"\n{BLUE}📋 Validation checks passed:{RESET}")
        print("  - Agent class inherits from BaseAgent")
        print("  - Agent implements all required methods (act, reset, seed)")
        print("  - Agent can be instantiated without arguments")
        print("  - Agent returns valid actions when given observations")
    
    print("\n" + "=" * 50)
    
    if results['valid']:
        print(f"\n{GREEN}🎉 Your agent is ready for submission!{RESET}")
        if filepath:
            print(f"{BLUE}Run this command to test performance before submitting:{RESET}")
            print(f"python src/evaluate_submission.py {filepath} --seeds 1 --num-seeds 10")
    else:
        print(f"\n{YELLOW}🔧 Please fix the errors above and validate again.{RESET}")
        print(f"{BLUE}Review the BaseAgent class in src/agents/base_agent.py for reference.{RESET}")


def main():
    """Main function to validate an agent."""
    args = parse_args()
    
    # Test with submission example if no arguments provided
    if args.agent_file is None:
        test_file = os.path.join(os.path.dirname(__file__), 'agents/agent_naive.py')
        print(f"{YELLOW}No agent file specified. Using example agent: {test_file}{RESET}")
        print(f"{YELLOW}Usage: python src/test_agent_validity.py path/to/your_agent.py{RESET}")
    else:
        test_file = args.agent_file
        if not os.path.exists(test_file):
            print(f"{RED}Error: File '{test_file}' not found.{RESET}")
            return 1
    
    print(f"{BLUE}✨ Validating agent in: {test_file} ✨{RESET}")
    print("=" * 50)
    
    # Validate the agent
    results = validate_agent(test_file)
    
    # Print results
    print_validation_results(results, args.verbose, test_file)
    
    # Return 0 if valid, 1 if not
    return 0 if results['valid'] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Validation cancelled by user.{RESET}")
        sys.exit(130) 