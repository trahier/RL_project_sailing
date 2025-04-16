#!/usr/bin/env python3
"""
Evaluate Sailing Agent Submission

This script evaluates a sailing agent on specified scenarios and reports performance metrics.
It can be used from the command line to quickly test agents without a notebook.

Usage:
    python evaluate_submission.py path/to/agent.py [options]

Options:
    --scenario SCENARIO    Name of scenario to evaluate on (default: all training scenarios)
    --seeds SEEDS          Space-separated list of seeds (default: 10 random seeds)
    --max_horizon N        Maximum steps per episode (default: 200)
    --render               Enable rendering (only works with a single seed)
    --output FILE          Save results to a JSON file
    --include-test         Include the hidden test scenario (evaluator use only)
    --verbose              Show detailed evaluation results
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('..'))

# Use relative imports
from test_agent_validity import validate_agent
from evaluation import evaluate_agent, visualize_trajectory

# Import scenarios with proper path handling
try:
    # Try direct import first (when running from src directory)
    from scenarios import get_scenario, SCENARIOS
    # Try to import test scenario if available (for evaluators only)
    try:
        from scenarios.private_scenarios import TEST_SCENARIO
        HAS_TEST_SCENARIO = True
    except (ImportError, FileNotFoundError):
        # Add a more user-friendly message
        HAS_TEST_SCENARIO = False
        TEST_SCENARIO = None
except ImportError:
    # If that fails, try importing from parent package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scenarios import get_scenario, SCENARIOS
    try:
        from scenarios.private_scenarios import TEST_SCENARIO
        HAS_TEST_SCENARIO = True
    except (ImportError, FileNotFoundError):
        # Add a more user-friendly message
        HAS_TEST_SCENARIO = False
        TEST_SCENARIO = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a sailing agent")
    
    parser.add_argument(
        "agent_path",
        type=str,
        help="Path to the agent implementation file"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Name of scenario to evaluate on (default: all training scenarios)"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated list of seeds (default: 10 random seeds)"
    )
    
    parser.add_argument(
        "--max_horizon",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (only works with a single seed)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a JSON file"
    )
    
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include the hidden test scenario (evaluator use only)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed evaluation results"
    )
    
    return parser.parse_args()


def print_results(scenario_name: str, results: Dict[str, Any], is_test: bool = False, verbose: bool = False):
    """Print evaluation results in a readable format."""
    if verbose:
        print(f"\nResults for scenario: {scenario_name}" + (" (HIDDEN TEST)" if is_test else ""))
        print("-" * 40)
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
        
        # Show more details for test scenario
        if is_test:
            print("\nIndividual results by seed:")
            for result in results['individual_results']:
                success = "✅" if result['success'] else "❌"
                print(f"  Seed {result['seed']}: {success} | Reward: {result['discounted_reward']:.2f} | Steps: {result['steps']}")
    else:
        # Simplified output
        scenario_label = f"{scenario_name}" + (" (TEST)" if is_test else "")
        print(f"{scenario_label:12} | Success: {results['success_rate']:.2%} | Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f} | Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")


def main():
    """Main evaluation function."""
    # Parse command line arguments
    args = parse_args()
    
    # Validate the agent
    print(f"Validating agent: {args.agent_path}")
    validation_results = validate_agent(args.agent_path)
    
    if not validation_results['valid']:
        print("❌ Agent validation failed:")
        for error in validation_results['errors']:
            print(f"  - {error}")
        sys.exit(1)
    
    # Create agent instance
    AgentClass = validation_results['agent_class']
    agent = AgentClass()
    print(f"✅ Successfully loaded agent: {AgentClass.__name__}")
    
    # Determine which scenarios to evaluate on
    if args.scenario:
        # Special handling for test scenario
        if args.scenario.lower() == "test":
            if not HAS_TEST_SCENARIO:
                print("❌ Error: Test scenario is not available. This feature is for evaluators only.")
                sys.exit(1)
            scenario_names = ["test"]
        else:
            # Single specified scenario
            scenario_names = [args.scenario]
    else:
        # All training scenarios by default
        scenario_names = [name for name in SCENARIOS.keys() if name.startswith("training_")]
    
    # Add test scenario if requested via --include-test
    if args.include_test:
        if not HAS_TEST_SCENARIO:
            print("❌ Error: Test scenario is not available. This feature is for evaluators only.")
            sys.exit(1)
        if "test" not in scenario_names:
            scenario_names.append("test")
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    else:
        # Generate 10 random seeds if not specified
        np.random.seed(42)  # For reproducibility
        seeds = np.random.randint(0, 1000, 10).tolist()
    
    # Check if rendering is compatible with seeds
    if args.render and len(seeds) > 1:
        print("⚠️ Rendering only works with a single seed. Using only the first seed.")
        seeds = [seeds[0]]
    
    # Store results for all scenarios
    all_results = {}
    
    # Print evaluation settings
    print(f"\nEvaluating on {len(scenario_names)} scenarios with {len(seeds)} seeds")
    print(f"Max horizon: {args.max_horizon} steps")
    
    # Header for simplified output
    if not args.verbose:
        print("\nSCENARIO     | SUCCESS RATE | MEAN REWARD       | MEAN STEPS")
        print("-" * 70)
    
    # Evaluate on each scenario
    for scenario_name in scenario_names:
        try:
            # Get the scenario
            if scenario_name.lower() == "test":
                if not HAS_TEST_SCENARIO:
                    raise ValueError("Test scenario is not available")
                scenario = TEST_SCENARIO.copy()
                is_test = True
            else:
                scenario = get_scenario(scenario_name)
                is_test = False
            
            # Add visualization parameters if rendering
            if args.render:
                scenario.update({
                    'env_params': {
                        'wind_grid_density': 25,
                        'wind_arrow_scale': 80,
                        'render_mode': "rgb_array"
                    }
                })
            
            # Run evaluation
            results = evaluate_agent(
                agent=agent,
                scenario=scenario,
                seeds=seeds,
                max_horizon=args.max_horizon,
                verbose=True,
                render=args.render,
                full_trajectory=args.render
            )
            
            # Store results
            all_results[scenario_name] = results
            
            # Print results
            print_results(scenario_name, results, is_test, args.verbose)
            
        except Exception as e:
            print(f"❌ Error evaluating on scenario {scenario_name}: {str(e)}")
    
    # Calculate overall performance
    if len(all_results) > 0:
        # Calculate standard metrics
        overall_success = sum(r['success_rate'] for r in all_results.values()) / len(all_results)
        overall_reward = sum(r['mean_reward'] for r in all_results.values()) / len(all_results)
        overall_steps = sum(r['mean_steps'] for r in all_results.values()) / len(all_results)
        
        # Calculate standard deviations of the means across scenarios
        reward_means = [r['mean_reward'] for r in all_results.values()]
        success_means = [r['success_rate'] for r in all_results.values()]
        steps_means = [r['mean_steps'] for r in all_results.values()]
        
        # When there's more than one scenario, calculate std across scenarios
        # Otherwise use the std from the single scenario
        if len(all_results) > 1:
            reward_std_of_means = np.std(reward_means) if len(reward_means) > 1 else 0
            success_std_of_means = np.std(success_means) if len(success_means) > 1 else 0
            steps_std_of_means = np.std(steps_means) if len(steps_means) > 1 else 0
        else:
            # For single scenario, use the scenario's own standard deviation
            scenario = list(all_results.values())[0]
            reward_std_of_means = scenario['std_reward']
            success_std_of_means = 0  # Success rate doesn't have a std in scenario results
            steps_std_of_means = scenario['std_steps']
        
        # Print summary
        print("\n" + "="*70)
        if args.verbose:
            print(f"OVERALL SUCCESS RATE: {overall_success:.2%} ± {success_std_of_means:.2%}")
            print(f"AVERAGE REWARD: {overall_reward:.2f} ± {reward_std_of_means:.2f}")
            print(f"AVERAGE STEPS: {overall_steps:.1f} ± {steps_std_of_means:.1f}")
            
            # If test scenario was included, calculate a weighted score (50% test, 50% training)
            if "test" in all_results:
                test_success = all_results["test"]["success_rate"]
                test_reward = all_results["test"]["mean_reward"]
                
                print("\nTEST SCENARIO PERFORMANCE:")
                print(f"  Success Rate: {test_success:.2%}")
                print(f"  Mean Reward: {test_reward:.2f}")
                
                # Calculate weighted score
                training_scenarios = [s for s in all_results.keys() if s != "test"]
                if training_scenarios:
                    training_success = sum(all_results[s]['success_rate'] for s in training_scenarios) / len(training_scenarios)
                    training_reward = sum(all_results[s]['mean_reward'] for s in training_scenarios) / len(training_scenarios)
                    weighted_reward = 0.5 * test_reward + 0.5 * training_reward
                    print(f"\nWEIGHTED FINAL REWARD: {weighted_reward:.2f}")
        else:
            # Simplified summary
            print(f"OVERALL      | {overall_success:.2%} ± {success_std_of_means:.2%} | {overall_reward:.2f} ± {reward_std_of_means:.2f} | {overall_steps:.1f} ± {steps_std_of_means:.1f}")
            
            # Add test info if included
            if "test" in all_results:
                test_reward = all_results["test"]["mean_reward"]
                training_scenarios = [s for s in all_results.keys() if s != "test"]
                if training_scenarios:
                    training_reward = sum(all_results[s]['mean_reward'] for s in training_scenarios) / len(training_scenarios)
                    weighted_reward = 0.5 * test_reward + 0.5 * training_reward
                    print(f"WEIGHTED FINAL REWARD: {weighted_reward:.2f} (50% test, 50% training)")
                    
        print("="*70)
        
        # Save results to file if requested
        if args.output:
            # Convert numpy values to Python native types for JSON serialization
            json_results = {
                "overall": {
                    "success_rate": float(overall_success),
                    "success_std_across_scenarios": float(success_std_of_means),
                    "average_reward": float(overall_reward),
                    "reward_std_across_scenarios": float(reward_std_of_means),
                    "average_steps": float(overall_steps),
                    "steps_std_across_scenarios": float(steps_std_of_means)
                },
                "scenarios": {}
            }
            
            for scenario_name, results in all_results.items():
                json_results["scenarios"][scenario_name] = {
                    "success_rate": float(results["success_rate"]),
                    "mean_reward": float(results["mean_reward"]),
                    "std_reward": float(results["std_reward"]),
                    "mean_steps": float(results["mean_steps"]),
                    "std_steps": float(results["std_steps"])
                }
            
            # Add weighted reward if test scenario was included
            if "test" in all_results and len(training_scenarios) > 0:
                json_results["weighted_reward"] = float(weighted_reward)
            
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main() 