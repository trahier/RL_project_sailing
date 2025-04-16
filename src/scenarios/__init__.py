"""
Predefined scenarios for the sailing challenge.
Each scenario presents different wind patterns and evolution characteristics.
"""

# Training Scenario 1: Clockwise Rotating Wind
# Characteristics: Predictable clockwise rotation with moderate speed variations
TRAINING_SCENARIO_1 = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (-0.65, -0.75),  # NNW wind
        'pattern_scale': 32,
        'pattern_strength': 0.3,
        'strength_variation': 0.5,
        'noise': 0.1
    },
    'wind_evol_params': {
        'wind_change_prob': 1,
        'pattern_scale': 128,
        'perturbation_angle_amplitude': 0.2,
        'perturbation_strength_amplitude': 0.2,
        'wind_evolution_bias': (1, 0.0),  # Eastward bias
        'bias_strength': 0.2
    }
}

# Training Scenario 2: Counter-Clockwise Rotation
# Characteristics: Counter-clockwise rotation with stronger variations
TRAINING_SCENARIO_2 = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (0.65, -0.75),  # NNE wind
        'pattern_scale': 32,
        'pattern_strength': 0.4,
        'strength_variation': 0.5,
        'noise': 0.1
    },
    'wind_evol_params': {
        'wind_change_prob': 1,
        'pattern_scale': 128,
        'perturbation_angle_amplitude': 0.2,
        'perturbation_strength_amplitude': 0.2,
        'wind_evolution_bias': (-1, 0.0),  # Westward bias
        'bias_strength': 0.2
    }
}

# Training Scenario 3: Oscillating Wind
# Characteristics: North-South oscillation with varying speeds
TRAINING_SCENARIO_3 = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (0.0, -1.0),  # Pure North wind
        'pattern_scale': 32,
        'pattern_strength': 0.3,
        'strength_variation': 0.4,
        'noise': 0.1
    },
    'wind_evol_params': {
        'wind_change_prob': 0.8,  # Slightly less frequent updates
        'pattern_scale': 96,      # Smaller scale than rotating scenarios
        'perturbation_angle_amplitude': 0.15,
        'perturbation_strength_amplitude': 0.3,  # More speed variation
        'wind_evolution_bias': (0.0, 0.0),  # No directional bias
        'bias_strength': 0.0      # Pure oscillation
    }
}

# Simple Test Scenario: Stable NE wind with minimal variations
# Characteristics: Very stable conditions, minimal noise, infrequent changes
SIMPLE_TEST_SCENARIO = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (-0.7, -0.7),  # NE wind
        'pattern_scale': 32,
        'pattern_strength': 0.1,     # Very small variations
        'strength_variation': 0.1,   # Very small variations
        'noise': 0.05               # Minimal noise
    },
    'wind_evol_params': {
        'wind_change_prob': 0.2,    # Infrequent changes
        'pattern_scale': 128,
        'perturbation_angle_amplitude': 0.05,
        'perturbation_strength_amplitude': 0.05,
        'wind_evolution_bias': (1.0, 0.0),  # Slight eastward bias
        'bias_strength': 0.1
    }
}

# Dictionary mapping scenario names to their parameters
SCENARIOS = {
    'training_1': TRAINING_SCENARIO_1,
    'training_2': TRAINING_SCENARIO_2,
    'training_3': TRAINING_SCENARIO_3,
    'simple_test': SIMPLE_TEST_SCENARIO
}

def get_scenario(name):
    """
    Get the parameters for a specific scenario.
    
    Args:
        name: String, one of ['training_1', 'training_2', 'training_3', 'simple_test']
        
    Returns:
        Dictionary containing wind_init_params and wind_evol_params
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {list(SCENARIOS.keys())}")
    return SCENARIOS[name] 