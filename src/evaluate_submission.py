#!/usr/bin/env python3
"""
Sailing Challenge - Agent Evaluation Script

This script evaluates a sailing agent on specified initial windfields and reports performance metrics.
"""

import argparse
import importlib.util
import numpy as np # type: ignore
import os
import sys
from typing import Dict, Any, List, Tuple, Optional

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base agent type
from agents.base_agent import BaseAgent

# Import evaluation functions
from evaluation import evaluate_agent

# Import initial windfields with proper path handling
try:
    from initial_windfields import get_initial_windfield, INITIAL_WINDFIELDS
    # Try to import test initial windfield if available (for evaluators only)
    try:
        from initial_windfields.private_initial_windfields import TEST_INITIAL_WINDFIELD
        HAS_TEST_INITIAL_WINDFIELD = True
    except ImportError:
        HAS_TEST_INITIAL_WINDFIELD = False
        TEST_INITIAL_WINDFIELD = None
except ImportError:
    # If we're running from the src directory
    from initial_windfields import get_initial_windfield, INITIAL_WINDFIELDS
    try:
        from initial_windfields.private_initial_windfields import TEST_INITIAL_WINDFIELD
        HAS_TEST_INITIAL_WINDFIELD = True
    except ImportError:
        HAS_TEST_INITIAL_WINDFIELD = False
        TEST_INITIAL_WINDFIELD = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a sailing agent on specified initial windfields.")
    
    parser.add_argument(
        "agent_file",
        help="Path to the Python file containing the agent implementation"
    )
    
    parser.add_argument(
        "--initial_windfield",
        choices=[*INITIAL_WINDFIELDS.keys(), "test"],
        help="Name of initial windfield to evaluate on (default: all training initial windfields)"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Combined with --num-seeds, specifies starting seed (default: 1)"
    )
    
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of evaluation seeds to use (default: 100)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed evaluation information"
    )
    
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Show seed information for progress tracking without full verbose output"
    )
    
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include the hidden test initial windfield (evaluator use only)"
    )
    
    return parser.parse_args()

def load_agent_from_file(file_path: str) -> BaseAgent:
    """Load agent class from a file dynamically."""
    # Get the absolute path
    abs_path = os.path.abspath(file_path)
    
    # Extract module name from file path (without .py extension)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the agent file at {abs_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find all classes that are subclasses of BaseAgent
    agent_classes = [
        cls for name, cls in module.__dict__.items()
        if isinstance(cls, type) and issubclass(cls, BaseAgent) and cls != BaseAgent
    ]
    
    if not agent_classes:
        raise ValueError(f"No valid agent classes found in {file_path}")
    
    # Get the first agent class
    agent_class = agent_classes[0]
    
    # Instantiate the agent
    agent = agent_class()
    
    return agent

def print_results(initial_windfield_name: str, results: Dict[str, Any], is_test: bool = False, verbose: bool = False):
    """Print the evaluation results in a readable format."""
    if verbose:
        # Print detailed results
        print(f"\nResults for initial windfield: {initial_windfield_name}" + (" (HIDDEN TEST)" if is_test else ""), flush=True)
        print(f"Success rate: {results['success_rate']:.2%}", flush=True)
        print(f"Discounted rewards mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}", flush=True)
        print(f"Step count mean: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}", flush=True)
        
        # Show more details for test initial windfield
        if is_test and 'individual_results' in results:
            print("\nIndividual results for test:", flush=True)
            for i, res in enumerate(results['individual_results']):
                print(f"Seed {res['seed']}: Reward = {res['discounted_reward']:.2f}, Steps = {res['steps']}, Success = {res['success']}", flush=True)
    else:
        # Print simple results
        initial_windfield_label = f"{initial_windfield_name}" + (" (TEST)" if is_test else "")
        print(f"{initial_windfield_label:12} | Success: {results['success_rate']:.2%} | Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f} | Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}", flush=True)

def weighted_score(training_results: Dict[str, float], test_result: Dict[str, float], weights: Tuple[float, float] = (0.5, 0.5)):
    """Calculate weighted score between training and test results."""
    training_weight, test_weight = weights
    
    # Combine success rates
    success_rate = (training_results['success_rate'] * training_weight + 
                   test_result['success_rate'] * test_weight)
    
    # Combine rewards
    reward = (training_results['reward'] * training_weight + 
             test_result['reward'] * test_weight)
    
    return {
        'success_rate': success_rate,
        'reward': reward
    }

def main():
    """Main function to evaluate an agent."""
    args = parse_args()
    
    try:
        # Load the agent from the specified file
        agent = load_agent_from_file(args.agent_file)
        print(f"Loaded agent: {type(agent).__name__}", flush=True)
        
        # Generate seeds for evaluation
        seeds = list(range(args.seeds, args.seeds + args.num_seeds))
        
        # Determine which initial windfields to evaluate on
        if args.initial_windfield:
            # Special handling for test initial windfield
            if args.initial_windfield.lower() == "test":
                if not HAS_TEST_INITIAL_WINDFIELD:
                    print("❌ Error: Test initial windfield is not available. This feature is for evaluators only.", flush=True)
                    return
                initial_windfield_names = ["test"]
            else:
                # Single specified initial windfield
                initial_windfield_names = [args.initial_windfield]
        else:
            # All training initial windfields by default
            initial_windfield_names = [name for name in INITIAL_WINDFIELDS.keys() if name.startswith("training_")]
            
        # Add test initial windfield if requested via --include-test
        if args.include_test:
            if not HAS_TEST_INITIAL_WINDFIELD:
                print("❌ Error: Test initial windfield is not available. This feature is for evaluators only.", flush=True)
                return
            if "test" not in initial_windfield_names:
                initial_windfield_names.append("test")
        
        # Basic statistical info about the evaluation
        np.set_printoptions(precision=2)
        
        # Set up default parameters for all evaluations
        eval_params = {
            'max_horizon': 500,  # Increased from 200 to 500
            'verbose': False,
            'render': False,
            'full_trajectory': False
        }
        
        # Store results for all initial windfields
        all_results = {}
        
        # Print evaluation parameters
        print(f"\nEvaluating on {len(initial_windfield_names)} initial windfields with {len(seeds)} seeds", flush=True)
        print(f"Agent: {type(agent).__name__}", flush=True)
        print(f"Maximum steps per episode: {eval_params['max_horizon']}", flush=True)
        
        # Print table header
        print("\nINITIAL_WINDFIELD     | SUCCESS RATE | MEAN REWARD       | MEAN STEPS", flush=True)
        print("-" * 75, flush=True)
        
        # Custom callback for seed-by-seed progress reporting
        def seed_callback(seed, results):
            """Callback function to report progress after each seed evaluation."""
            if args.show_seeds or args.verbose:
                print(f"Seed {seed}: Reward = {results['discounted_reward']:.2f}, Steps = {results['steps']}, Success = {results['success']}", flush=True)
        
        # Set verbose parameter if show_seeds is enabled
        if args.show_seeds and not args.verbose:
            print_seed_info = True
        else:
            print_seed_info = args.verbose
        
        # Evaluate on each initial windfield
        for initial_windfield_name in initial_windfield_names:
            try:
                # Get the initial windfield
                if initial_windfield_name.lower() == "test":
                    if not HAS_TEST_INITIAL_WINDFIELD:
                        raise ValueError("Test initial windfield is not available")
                    initial_windfield = TEST_INITIAL_WINDFIELD.copy()
                    is_test = True
                else:
                    initial_windfield = get_initial_windfield(initial_windfield_name)
                    is_test = False
                
                # Add render mode to environment parameters if not specified
                initial_windfield.update({
                    'env_params': {
                        **initial_windfield.get('env_params', {}),
                        'render_mode': None
                    }
                })
                
                # Run evaluation
                results = evaluate_agent(
                    agent=agent,
                    initial_windfield=initial_windfield,
                    seeds=seeds,
                    seed_callback=seed_callback if print_seed_info else None,
                    **eval_params
                )
                
                # Store results
                all_results[initial_windfield_name] = results
                
                # Print results
                print_results(initial_windfield_name, results, is_test, args.verbose)
                
            except Exception as e:
                print(f"❌ Error evaluating on initial windfield {initial_windfield_name}: {str(e)}", flush=True)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print("-" * 75, flush=True)
        
        # Calculate combined statistics
        if len(all_results) > 0:
            # Calculate means across all initial windfields
            success_rates = np.array([r['success_rate'] for r in all_results.values()])
            rewards = np.array([r['mean_reward'] for r in all_results.values()])
            steps = np.array([r['mean_steps'] for r in all_results.values()])
            
            # Calculate standard deviations of the means across initial windfields
            success_std_of_means = np.std(success_rates) if len(success_rates) > 1 else 0
            reward_std_of_means = np.std(rewards) if len(rewards) > 1 else 0
            steps_std_of_means = np.std(steps) if len(steps) > 1 else 0
            
            # When there's only one initial windfield evaluated
            if len(all_results) == 1:
                # For single initial windfield, use the initial windfield's own standard deviation
                initial_windfield = list(all_results.values())[0]
                reward_std_of_means = initial_windfield['std_reward']
                success_std_of_means = 0  # Success rate doesn't have a std in initial windfield results
                steps_std_of_means = initial_windfield['std_steps']
            
            # Print overall results
            print(f"OVERALL      | Success: {np.mean(success_rates):.2%} ± {success_std_of_means:.2%}", flush=True)
            print(f"Reward: {np.mean(rewards):.2f} ± {reward_std_of_means:.2f}", flush=True)
            print(f"Steps: {np.mean(steps):.1f} ± {steps_std_of_means:.1f}", flush=True)
            
            # If test initial windfield was included, calculate a weighted score (50% test, 50% training)
            if 'test' in all_results and len(all_results) > 1:
                test_results = all_results['test']
                
                # Calculate mean of training results
                training_success = np.mean([r['success_rate'] for name, r in all_results.items() if name != 'test'])
                training_reward = np.mean([r['mean_reward'] for name, r in all_results.items() if name != 'test'])
                
                print("\nTEST INITIAL WINDFIELD PERFORMANCE:", flush=True)
                print(f"Training success rate: {training_success:.2%}", flush=True)
                print(f"Test: {test_results['success_rate']:.2%}", flush=True)
                print(f"Training avg reward: {training_reward:.2f}", flush=True)
                print(f"Test: {test_results['mean_reward']:.2f}", flush=True)
                
                # Calculate weighted score for reporting
                weighted = {
                    'success_rate': 0.5 * training_success + 0.5 * test_results['success_rate'],
                    'reward': 0.5 * training_reward + 0.5 * test_results['mean_reward']
                }
                
                print(f"\nFINAL SCORE (50% training, 50% test):", flush=True)
                print(f"  Success rate: {weighted['success_rate']:.2%}", flush=True)
                print(f"  Average reward: {weighted['reward']:.2f}", flush=True)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 