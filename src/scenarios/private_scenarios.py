"""
Private scenarios for the evaluator of the Sailing Challenge.
This file contains scenarios that are not shared with students.
"""

# Import from the scenarios package
from scenarios import get_scenario, SCENARIOS

# Test Scenario: Complex Evolution
# This scenario combines elements from all training scenarios but with unique characteristics
# It should be challenging but manageable for agents that learned from the training scenarios
TEST_SCENARIO = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (-0.3, -0.95),  # Slightly off-North
        'pattern_scale': 32,
        'pattern_strength': 0.35,         # Between training scenarios
        'strength_variation': 0.45,
        'noise': 0.1
    },
    'wind_evol_params': {
        'wind_change_prob': 0.9,          # Frequent but not constant
        'pattern_scale': 112,             # Between training scales
        'perturbation_angle_amplitude': 0.18,  # Moderate angle changes
        'perturbation_strength_amplitude': 0.25,  # Moderate strength changes
        'wind_evolution_bias': (0.5, 0.0),     # Slight eastward bias
        'bias_strength': 0.15             # Moderate bias
    }
}

# Extend the scenarios dictionary with private scenarios
# Create a new dictionary to avoid modifying the original
ALL_SCENARIOS = SCENARIOS.copy()
ALL_SCENARIOS['test'] = TEST_SCENARIO

# We can use the original get_scenario function, but we provide a wrapper
# that can access both public and private scenarios
def get_test_scenario():
    """
    Get the private test scenario.
    
    Returns:
        Dictionary containing wind_init_params and wind_evol_params for the test scenario
    """
    return TEST_SCENARIO 