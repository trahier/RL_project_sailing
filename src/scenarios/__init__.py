"""
Predefined scenarios for the sailing challenge.
Each scenario presents different wind patterns and evolution characteristics.
"""

# Common wind evolution parameters
# Using the same wind evolution parameters for all scenarios for pedagogical purposes
# Wind will make a complete 180° rotation in ~100 steps on average with these parameters
COMMON_WIND_EVOL_PARAMS = {
    'wind_change_prob': 1.0,      # Wind field updates on every step
    'pattern_scale': 64,         # Scale of spatial perturbation patterns
    'perturbation_angle_amplitude': 0.12,  # Angle perturbation per step
    'perturbation_strength_amplitude': 0.15,  # Strength variation per step
    'rotation_bias': 0.02,        # Clockwise rotational bias (positive = clockwise)
    'bias_strength': 1.0          # Full strength of rotational bias
}

# Training Scenario 1: North-Northwest Wind
# Characteristics: Starting with NNW wind
TRAINING_SCENARIO_1 = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (-0.7, -0.7),  # NNW wind
        'pattern_scale': 32,
        'pattern_strength': 0.3,
        'strength_variation': 0.4,
        'noise': 0.1
    },
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Training Scenario 2: North-Northeast Wind
# Characteristics: Starting with NNE wind
TRAINING_SCENARIO_2 = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (0, -1), 
        'pattern_scale': 128,
        'pattern_strength': 0.7,
        'strength_variation': 0.2,
        'noise': 0.1
    },
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Training Scenario 3: Pure North Wind
# Characteristics: Starting with N wind, smaller pattern scale
TRAINING_SCENARIO_3 = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (0.9, 0.2),  
        'pattern_scale': 16,           
        'pattern_strength': 0.1,
        'strength_variation': 0.4,
        'noise': 0.1
    },
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Simple Static Scenario: Stable NE wind with NO variations
# Characteristics: Completely static conditions, minimal noise, no wind changes
SIMPLE_STATIC_SCENARIO = {
    'wind_init_params': {
        'base_speed': 4.0,
        'base_direction': (-0.7, -0.7),  # NE wind
        'pattern_scale': 32,
        'pattern_strength': 0.1,     # Very small variations
        'strength_variation': 0.1,   # Very small variations
        'noise': 0.05               # Minimal noise
    },
    'wind_evol_params': {
        'wind_change_prob': 0.0,    # No changes - completely static
        'pattern_scale': 128,
        'perturbation_angle_amplitude': 0.0,  # No angle perturbations
        'perturbation_strength_amplitude': 0.0,  # No strength perturbations
        'rotation_bias': 0.0,        # No rotational bias
        'bias_strength': 0.0         # No bias strength
    }
}

# Dictionary mapping scenario names to their parameters
SCENARIOS = {
    'training_1': TRAINING_SCENARIO_1,
    'training_2': TRAINING_SCENARIO_2,
    'training_3': TRAINING_SCENARIO_3,
    'simple_static': SIMPLE_STATIC_SCENARIO
}

def get_scenario(name):
    """
    Get the parameters for a specific scenario.
    
    Args:
        name: String, one of ['training_1', 'training_2', 'training_3', 'simple_static']
        
    Returns:
        Dictionary containing wind_init_params and wind_evol_params
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {list(SCENARIOS.keys())}")
    return SCENARIOS[name] 