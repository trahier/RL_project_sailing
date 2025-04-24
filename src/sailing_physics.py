"""
Sailing Physics Visualizations

This module provides functions to visualize the sailing physics used in the environment.
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def generate_velocity_polar_plot(wind_strength=1.0, boat_performance=0.4):
    """
    Generate a polar plot showing boat velocity as a function of angle to wind.
    
    Args:
        wind_strength: Wind speed (default 1.0)
        boat_performance: Boat performance factor (default 0.4)
        
    Returns:
        fig: Matplotlib figure containing the polar plot
    """
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Create angles from 0 to 360 degrees
    angles = np.linspace(0, 2*np.pi, 360)
    
    # Calculate boat speed at each angle
    speeds = []
    for angle in angles:
        # Create normalized vectors for boat direction and wind
        boat_direction = np.array([np.sin(angle), np.cos(angle)])
        wind_direction = np.array([0.0, -1.0])  # Wind pointing down
        
        # Calculate efficiency using the shared function
        efficiency = calculate_sailing_efficiency(boat_direction, wind_direction)
        
        # Calculate speed
        speed = efficiency * wind_strength * boat_performance
        speeds.append(speed)
    
    # Plot boat speed
    ax.plot(angles, speeds, 'r-', linewidth=2, label='Boat Speed')
    
    # Plot wind direction (inverted to show wind coming FROM the top)
    # Move the arrow higher in the plot (start at y=0.8, end at y=0.3)
    ax.arrow(0, 0.8, 0, -0.5, color='blue', width=0.02, head_width=0.1, 
             head_length=0.1, transform=ax.transData._b, zorder=3, label='Wind Direction')
    
    # Add labels for sailing zones
    ax.text(np.pi/4, 1.2, 'Close-Hauled', color='green', ha='left', va='bottom')
    ax.text(np.pi/2, 1.2, 'Beam Reach', color='green', ha='left', va='bottom')
    ax.text(3*np.pi/4, 1.2, 'Broad Reach', color='green', ha='left', va='bottom')
    ax.text(0, 1.4, 'Upwind\n(No-Go Zone)', color='red', ha='center', va='bottom')
    ax.text(np.pi, 1.2, 'Downwind', color='green', ha='center', va='bottom')
    
    # Shade the low efficiency zone
    low_eff_angles = np.linspace(-np.pi/4, np.pi/4, 100)
    ax.fill_between(low_eff_angles, 0, 1.5, color='red', alpha=0.2, label='Low Efficiency Zone')
    
    # Set up the plot
    ax.set_theta_zero_location('N')  # 0 degrees at the top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_rmax(1.5)
    ax.set_title(f'Boat Velocity Polar Diagram (Wind Strength: {wind_strength}, Performance: {boat_performance})')
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    ax.grid(True)
    
    return fig

def generate_efficiency_curve():
    """
    Generate a plot showing sailing efficiency as a function of wind angle.
    Angle of 0 degrees means sailing directly into the wind (headwind).
    Angle of 180 degrees means sailing with the wind (tailwind).
    """
    # Create angles from 0 to 180 degrees (0 is facewind, 180 is tailwind)
    angles = np.linspace(0, np.pi, 100)
    
    # Create wind vector (pointing down)
    wind_direction = np.array([0.0, -1.0])
    
    # Calculate efficiency for each angle
    efficiencies = []
    for angle in angles:
        # Create boat direction vector at this angle
        # Using -angle because 0 should be pointing up (against wind)
        boat_direction = np.array([np.sin(-angle), np.cos(-angle)])
        # Calculate efficiency using the shared function
        efficiency = calculate_sailing_efficiency(boat_direction, wind_direction)
        efficiencies.append(efficiency)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the efficiency curve
    ax.plot(np.degrees(angles), efficiencies, linewidth=2)
    
    # Add colored regions
    ax.fill_between(np.degrees(angles[0:25]), 0, efficiencies[0:25], color='red', alpha=0.2, label='No-Go Zone')
    ax.fill_between(np.degrees(angles[25:50]), 0, efficiencies[25:50], color='orange', alpha=0.2, label='Close-Hauled')
    ax.fill_between(np.degrees(angles[50:75]), 0, efficiencies[50:75], color='green', alpha=0.2, label='Beam Reach')
    ax.fill_between(np.degrees(angles[75:]), 0, efficiencies[75:], color='blue', alpha=0.2, label='Broad Reach')
    
    # Add angle markers
    ax.axvline(x=45, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=90, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=135, color='b', linestyle='--', alpha=0.5)
    
    # Add labels for different sailing zones
    ax.text(22.5, 1.05, 'No-Go Zone', ha='center')
    ax.text(67.5, 1.05, 'Close-Hauled', ha='center')
    ax.text(112.5, 1.05, 'Beam Reach', ha='center')
    ax.text(157.5, 1.05, 'Broad Reach', ha='center')
    
    # Set labels and limits
    ax.set_xlabel('Angle to Wind (degrees)')
    ax.set_ylabel('Sailing Efficiency')
    ax.set_title('Sailing Efficiency vs. Wind Angle')
    ax.grid(True)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 180)
    
    # Add legend
    ax.legend(loc='lower right')
    
    return fig

def show_tacking_maneuver():
    """
    Demonstrate tacking (zig-zagging) as the strategy for sailing upwind.
    
    Returns:
        fig: The matplotlib figure showing the tacking maneuver
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define the starting point and destination
    # Start at bottom, destination at top (upwind)
    start = np.array([100, 20])
    dest = np.array([100, 180])
    
    # Plot the start, destination and direct path
    ax.scatter(start[0], start[1], s=100, color='green', label='Start')
    ax.scatter(dest[0], dest[1], s=100, color='red', label='Destination (Upwind)')
    ax.plot([start[0], dest[0]], [start[1], dest[1]], 'k--', label='Direct Route (Impossible)')
    
    # Define the tacking path with realistic zigzag
    # Each segment should be at approximately 45° to the wind (close-hauled)
    tack_points = [
        start,
        np.array([130, 50]),    # First tack to the right
        np.array([70, 80]),     # Tack to the left
        np.array([130, 110]),   # Tack to the right
        np.array([70, 140]),    # Tack to the left
        np.array([130, 170]),   # Tack to the right
        dest
    ]
    
    # Plot the tacking path
    tack_x = [p[0] for p in tack_points]
    tack_y = [p[1] for p in tack_points]
    ax.plot(tack_x, tack_y, 'g-', linewidth=2, label='Tacking Route')
    
    # Plot wind direction indicators - pointing downward (from top to bottom)
    wind_positions = np.mgrid[60:140:20, 40:180:20].reshape(2, -1).T
    for pos in wind_positions:
        ax.arrow(pos[0], pos[1], 0, -10, head_width=3, head_length=5, fc='blue', ec='blue', alpha=0.5)
    
    # Add an arrow showing wind direction near the text
    wind_arrow_pos = np.array([30, 100])
    ax.arrow(wind_arrow_pos[0], wind_arrow_pos[1], 0, -20, 
             head_width=6, head_length=10, fc='blue', ec='blue')
    ax.text(wind_arrow_pos[0] - 5, wind_arrow_pos[1] + 30, "Wind\nDirection", 
           fontsize=12, color='blue', ha='right')
    
    # Add a clearer tacking annotation
    ax.annotate('Tacking Pattern\n(Zig-Zag)', 
                xy=(100, 100), xytext=(50, 100),
                arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
                fontsize=12, color='green')
    
    # Set axis limits and labels
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Why sailors must \'tack\' (zig-zag) when sailing upwind', fontsize=14)
    
    # Add a grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    return fig 

def calculate_sailing_efficiency(boat_direction, wind_direction):
    """
    Calculate sailing efficiency based on the angle between boat direction and wind.
    
    Args:
        boat_direction: Normalized vector of boat's desired direction
        wind_direction: Normalized vector of wind direction (where wind is going TO)
        
    Returns:
        sailing_efficiency: Float between 0.05 and 1.0 representing how efficiently the boat can sail
    """
    # Invert wind direction to get where wind is coming FROM
    wind_from = -wind_direction
    
    # Calculate angle between wind and direction
    wind_angle = np.arccos(np.clip(
        np.dot(wind_from, boat_direction), -1.0, 1.0))
    
    # Calculate sailing efficiency based on angle to wind
    if wind_angle < np.pi/4:  # Less than 45 degrees to wind
        sailing_efficiency = 0.05  # Small but non-zero efficiency in no-go zone
    elif wind_angle < np.pi/2:  # Between 45 and 90 degrees
        sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)  # Linear increase to 1.0
    elif wind_angle < 3*np.pi/4:  # Between 90 and 135 degrees
        sailing_efficiency = 1.0  # Maximum efficiency
    else:  # More than 135 degrees
        sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4)  # Linear decrease
        sailing_efficiency = max(0.5, sailing_efficiency)  # But still decent
    
    return sailing_efficiency 