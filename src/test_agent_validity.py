"""
Test Agent Validity

This module provides functions to test if a sailing agent meets the required
interface specifications and behaves correctly.
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Type

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('..'))

# Import with proper path handling
from agents.base_agent import BaseAgent
from env_sailing import SailingEnv

# Try different approaches to import initial_windfields
try:
    # Direct import from the package
    from initial_windfields import get_initial_windfield
except ImportError:
    # Relative import in case the script is run from the src directory
    from initial_windfields import get_initial_windfield


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
            print(f"Warning: Multiple agent classes found in {filepath}. "
                  f"Using {agent_classes[0].__name__}")
        
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
                    f"Agent returned non-integer action: {action} (type: {type(action)})"
                )
                break
            
            # Check if action is in valid range
            if action < 0 or action > 8:
                issues.append(
                    f"Agent returned out-of-range action: {action} (valid range: 0-8)"
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
            results['errors'].append(f"Failed to instantiate agent: {str(e)}")
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


if __name__ == "__main__":
    # Test with submission example if no arguments provided
    if len(sys.argv) < 2:
        test_file = os.path.join(os.path.dirname(__file__), 'agents/agent_naive.py')
    else:
        test_file = sys.argv[1]
    
    # Validate the agent
    results = validate_agent(test_file)
    
    # Print results
    print(f"Agent validation results for: {test_file}")
    print(f"Agent name: {results['agent_name']}")
    print(f"Valid: {results['valid']}")
    
    if results['errors']:
        print("\nErrors:")
        for i, error in enumerate(results['errors'], 1):
            print(f"{i}. {error}")
    
    if results['warnings']:
        print("\nWarnings:")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"{i}. {warning}")
            
    if results['valid']:
        print("\nValidation successful! The agent meets all requirements.")
    else:
        print("\nValidation failed. Please fix the errors and try again.") 