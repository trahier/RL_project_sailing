# Wind Evolution System Documentation

## Overview
The sailing environment features a sophisticated wind system with two main components:
1. Initial wind field generation
2. Wind evolution over time

## Initial Wind Field Parameters (`wind_init_params`)

### Base Wind Configuration
- `base_speed` (default: 4.0)
  - Sets the reference wind speed for the entire field
  - All wind vectors will be normalized and scaled to approximately this magnitude
  - Higher values make the boat move faster overall

- `base_direction` (default: (-0.65, -0.75))
  - Sets the predominant wind direction
  - Format: (x, y) vector, will be normalized internally
  - Example: (-0.65, -0.75) creates a north-northwestern wind
  - Positive y is northward, positive x is eastward

### Spatial Variation Parameters
- `pattern_scale` (default: 32)
  - Controls the size of anticyclonic patterns in the wind field
  - Larger values create broader, more gradual variations
  - Should typically match the grid size for one complete pattern
  - Example: 32 for a 32x32 grid creates one complete rotation

- `pattern_strength` (default: 0.3)
  - Controls how much wind directions deviate from base_direction
  - Range: 0.0 to 1.0
  - Higher values create more pronounced rotations
  - 0.0 would give uniform direction across the field

- `strength_variation` (default: 0.4)
  - Controls spatial variation in wind speed
  - Applied as a multiplier: (1.0 ± strength_variation) * base_speed
  - Higher values create more pronounced speed differences
  - Example: 0.4 allows speeds from 0.6x to 1.4x base_speed

- `noise` (default: 0.1)
  - Adds random perturbations to both direction and speed
  - Applied after all other patterns
  - Helps prevent perfectly uniform or predictable patterns
  - Small values (0.05-0.15) recommended for natural appearance

## Wind Evolution Parameters (`wind_evol_params`)

### Evolution Control
- `wind_change_prob` (default: 0.1)
  - Probability of wind field update each step
  - 1.0 means update every step
  - 0.1 means update roughly every 10 steps
  - Higher values create more dynamic conditions

- `pattern_scale` (default: 32)
  - Controls the spatial scale of wind evolution patterns
  - Larger values create broader, more coordinated changes
  - Can differ from initial pattern_scale
  - Example: 128 creates very broad, sweeping changes

### Perturbation Parameters
- `perturbation_angle_amplitude` (default: 0.1)
  - Controls the magnitude of direction changes
  - Higher values create more dramatic direction shifts
  - Applied smoothly across the field
  - Recommended range: 0.1-0.3

- `perturbation_strength_amplitude` (default: 0.2)
  - Controls the magnitude of speed changes
  - Applied as a multiplier to base_speed
  - Independent of direction changes
  - Recommended range: 0.1-0.3

### Evolution Bias
- `wind_evolution_bias` (default: (0.0, 0.0))
  - Adds a preferred direction to wind evolution
  - Format: (x, y) vector, will be normalized internally
  - (0.0, 0.0) means no preferred direction
  - Example: (1.0, 0.0) biases evolution eastward

- `bias_strength` (default: 0.1)
  - Controls how strongly the bias affects evolution
  - Multiplied by perturbation amplitudes
  - Higher values make the bias more noticeable
  - Recommended range: 0.1-0.3

## Notable Scenarios

### Scenario 1: Clockwise Rotating Wind
```python
wind_init_params = {
    'base_speed': 4.0,
    'base_direction': (-0.65, -0.75),
    'pattern_scale': 32,
    'pattern_strength': 0.3,
    'strength_variation': 0.5,
    'noise': 0.1
}

wind_evol_params = {
    'wind_change_prob': 1,
    'pattern_scale': 128,
    'perturbation_angle_amplitude': 0.2,
    'perturbation_strength_amplitude': 0.2,
    'wind_evolution_bias': (1, 0.0),
    'bias_strength': 0.2
}
```
This scenario creates a wind field that gradually rotates clockwise, with:
- Frequent updates (every step)
- Large-scale evolution patterns
- Eastward evolution bias
- Moderate strength variations

### Scenario 2: Counter-Clockwise Rotation
```python
wind_init_params = {
    'base_speed': 4.0,
    'base_direction': (0.65, -0.75),
    'pattern_scale': 32,
    'pattern_strength': 0.4,
    'strength_variation': 0.5,
    'noise': 0.1
}

wind_evol_params = {
    'wind_change_prob': 1,
    'pattern_scale': 128,
    'perturbation_angle_amplitude': 0.2,
    'perturbation_strength_amplitude': 0.2,
    'wind_evolution_bias': (-1, 0.0),
    'bias_strength': 0.2
}
```
This scenario produces:
- Counter-clockwise rotation
- Stronger initial pattern variations
- Westward evolution bias
- Similar update frequency and perturbation scales

## Tips for Creating Scenarios
1. Match pattern_scale to grid size for initial patterns
2. Use larger pattern_scale for evolution (3-4x grid size) for smooth changes
3. Keep perturbation amplitudes below 0.3 to avoid chaotic behavior
4. Use bias_strength to create predictable trends in wind evolution
5. Adjust wind_change_prob based on desired update frequency 