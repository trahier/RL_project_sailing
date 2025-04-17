import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from typing import Tuple, Dict, Any, Optional
from src.sailing_physics import calculate_sailing_efficiency
from src.scenarios import SIMPLE_STATIC_SCENARIO

class SailingEnv(gym.Env):
    """
    A sailing navigation environment where an agent must navigate from
    a starting point to a destination while accounting for wind.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Default wind parameters - match the simple static scenario
    DEFAULT_WIND_INIT_PARAMS = SIMPLE_STATIC_SCENARIO['wind_init_params']
    DEFAULT_WIND_EVOL_PARAMS = SIMPLE_STATIC_SCENARIO['wind_evol_params']
    
    def __init__(self, 
                 grid_size=(32, 32),
                 wind_init_params=None,
                 wind_evol_params=None,
                 wind_grid_density=25,
                 wind_arrow_scale=100,
                 render_mode=None,
                 boat_performance=0.4,
                 max_speed=2.0,
                 inertia_factor=0.5,
                 reward_discount_factor=0.99):
        """
        Initialize the sailing environment.
        
        Args:
            grid_size: Tuple of (width, height) for the grid
            wind_init_params: Dictionary of wind initialization parameters
            wind_evol_params: Dictionary of wind evolution parameters
            wind_grid_density: Number of wind arrows to display (default: 25)
            wind_arrow_scale: Scale factor for wind arrow visualization (default: 100)
            render_mode: How to render the environment
            boat_performance: How well the boat converts wind to movement
            max_speed: Maximum boat speed
            inertia_factor: How much velocity is preserved (0-1)
            reward_discount_factor: Discount factor for future rewards
        """
        super().__init__()
        
        # Store parameters
        self.grid_size = grid_size
        self.wind_init_params = wind_init_params or self.DEFAULT_WIND_INIT_PARAMS.copy()
        self.wind_evol_params = wind_evol_params or self.DEFAULT_WIND_EVOL_PARAMS.copy()
        self.wind_grid_density = wind_grid_density
        self.wind_arrow_scale = wind_arrow_scale
        self.render_mode = render_mode
        self.boat_performance = boat_performance
        self.max_speed = max_speed
        self.inertia_factor = inertia_factor
        self.reward_discount_factor = reward_discount_factor
        
        # Initialize wind field
        self.wind_field = self._generate_wind_field()
        
        # Initialize boat state
        self.position = np.array([grid_size[0] // 2, 0])  # Start at bottom center
        self.velocity = np.array([0.0, 0.0])
        self.position_accumulator = np.array([0.0, 0.0])  # Initialize position accumulator
        self.goal_position = np.array([grid_size[0] // 2, grid_size[1] - 1])  # Goal at top center
        
        # Initialize step count
        self.step_count = 0
        
        # Initialize last action
        self.last_action = None
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(9)  # 0-7: Move in direction, 8: Stay in place
        
        # Calculate the shape for the full wind field (grid_size[0] x grid_size[1] x 2)
        wind_field_shape = (grid_size[0] * grid_size[1] * 2,)
        
        # Define observation space to include the full wind field
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6 + wind_field_shape[0],),  # [x, y, vx, vy, wx, wy, flattened wind field]
            dtype=np.float32
        )
        
        # Initialize random number generator
        self.np_random = None
        self.seed()
        
    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        self.np_random = np.random.default_rng(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Initialize or reset the random number generator
        self.seed(seed)
        
        # Reset position to bottom center
        self.position = np.array([self.grid_size[0] // 2, 0])
        
        # Reset velocity and position accumulator
        self.velocity = np.array([0.0, 0.0])
        self.position_accumulator = np.array([0.0, 0.0])
        
        # Reset step count and last action
        self.step_count = 0
        self.last_action = None
        
        # Generate new wind field
        self.wind_field = self._generate_wind_field()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Get info
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'wind': self._get_wind_at_position(self.position),
            'step': self.step_count
        }
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Integer in [0, 8] representing the action to take, or None to skip
            
        Returns:
            observation: Dictionary containing the new observation
            reward: Float reward signal
            terminated: Boolean indicating if the episode is over
            truncated: Boolean indicating if the episode was artificially terminated
            info: Dictionary containing additional information
        """
        # Store the action
        self.last_action = action
        
        # If action is None, return current state without advancing
        if action is None:
            return (
                self._get_observation(),
                0.0,  # No reward for waiting
                False,  # Not terminated
                False,  # Not truncated
                {}  # No additional info
            )
            
        self.step_count += 1
        
        # Convert action to direction
        direction = self._action_to_direction(action)
        
        # Get current wind at boat's position
        current_wind = self._get_wind_at_position(self.position)
        
        # Calculate new velocity based on sailing physics
        self.velocity = self._calculate_new_velocity(
            current_velocity=self.velocity,
            wind=current_wind,
            direction=direction
        )
        
        # Update position accumulator with sub-grid movements
        self.position_accumulator += self.velocity
        
        # Calculate new position including accumulated movement
        new_position_float = self.position + self.position_accumulator
        
        # Round to nearest grid cell
        new_position = np.round(new_position_float).astype(np.int32)
        
        # Reset accumulator but keep the remainder
        self.position_accumulator = new_position_float - new_position
        
        # Ensure position stays within bounds
        new_position = np.clip(
            new_position, 
            [0, 0], 
            [self.grid_size[0]-1, self.grid_size[1]-1]
        )
        
        # Update position
        self.position = new_position
        
        # Check if reached goal (within 1 cell)
        distance_to_goal = np.linalg.norm(self.position - self.goal_position)
        reached_goal = distance_to_goal < 1.5
        
        # Calculate reward
        reward = self._calculate_reward(reached_goal, distance_to_goal)
        
        # Determine if episode is done
        terminated = reached_goal or self.step_count >= 1000  # Increased from 200 to 1000
        truncated = False  # We don't truncate episodes
        
        # Possibly update wind field based on wind_change_prob
        if self.np_random.random() < self.wind_evol_params['wind_change_prob']:
            self._update_wind_field()
        
        # Create observation
        observation = self._get_observation()
        
        # Set info dictionary
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'wind': current_wind,
            'step_count': self.step_count,
            'distance_to_goal': distance_to_goal
        }
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        else:
            return self._render_frame()  # Default to rgb_array mode
    
    def _render_frame(self):
        """
        Render the current state as a frame.
        Uses the environment's wind_grid_density and wind_arrow_scale parameters.
        """
        # Create a figure for visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set the limits with a bit of padding
        ax.set_xlim(-1, self.grid_size[0] + 1)
        ax.set_ylim(-1, self.grid_size[1] + 1)
        
        # Create ocean-like background
        ocean_color = '#6BAED6'  # Medium blue color
        ax.set_facecolor(ocean_color)
        
        # Add a subtle grid to represent ocean waves
        ax.grid(True, linestyle='-', alpha=0.2, color='white')
        
        # Plot the wind field (using quiver for vector field)
        # Sample based on wind_grid_density
        step = max(1, self.grid_size[0] // self.wind_grid_density)
        x = np.arange(0, self.grid_size[0], step)
        y = np.arange(0, self.grid_size[1], step)
        X, Y = np.meshgrid(x, y)
        
        # Get wind vectors at sampled positions
        U = self.wind_field[::step, ::step, 0]  # x component
        V = self.wind_field[::step, ::step, 1]  # y component
        
        # Plot wind vectors with specified scale
        wind_arrows = ax.quiver(X, Y, U, V, color='white', alpha=0.6, scale=self.wind_arrow_scale)
        
        # Draw the boat as a triangle (sailboat from above)
        boat_direction = self.velocity
        if np.linalg.norm(boat_direction) < 0.1:
            # Default orientation - pointing toward the goal
            # Using the convention that North (goal) is (0, 1) increasing Y
            boat_direction = np.array([0, 1])
        else:
            boat_direction = boat_direction / np.linalg.norm(boat_direction)
            
        # Create a better boat shape (more boat-like triangle)
        boat_length = 1.8
        boat_width = 1.0
        
        # Define the boat points (bow at front, wider stern at back)
        bow = self.position + boat_direction * boat_length * 0.6
        stern_center = self.position - boat_direction * boat_length * 0.4
        port = stern_center + np.array([-boat_direction[1], boat_direction[0]]) * boat_width * 0.5
        starboard = stern_center + np.array([boat_direction[1], -boat_direction[0]]) * boat_width * 0.5
        
        # Create a polygon for the boat
        boat_vertices = np.array([bow, port, stern_center, starboard])
        boat_polygon = plt.Polygon(boat_vertices, color='#D62728', alpha=0.9)  # Brighter red
        ax.add_patch(boat_polygon)
        
        # Add velocity vector from the bow of the boat
        if np.linalg.norm(self.velocity) > 0.2:
            velocity_arrow = ax.arrow(
                bow[0], bow[1],
                self.velocity[0], self.velocity[1],
                head_width=0.3, head_length=0.3, fc='yellow', ec='yellow', alpha=0.8
            )
        else:
            velocity_arrow = None
        
        # Draw the goal (green circle with subtle glow effect)
        goal_outer = plt.Circle((self.goal_position[0], self.goal_position[1]), 1.5, color='green', alpha=0.3)
        goal_inner = plt.Circle((self.goal_position[0], self.goal_position[1]), 1.0, color='#2CA02C', alpha=0.7)  # Darker green
        ax.add_patch(goal_outer)
        ax.add_patch(goal_inner)
        
        # Add text information with better styling
        wind_at_pos = self._get_wind_at_position(self.position)
        action_names = {
            0: "North",
            1: "Northeast",
            2: "East",
            3: "Southeast", 
            4: "South",
            5: "Southwest",
            6: "West",
            7: "Northwest"
        }
        action_str = action_names.get(self.last_action, "None")
        info_text = (
            f"Step: {self.step_count}\n"
            f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f})\n"
            f"Velocity: ({self.velocity[0]:.2f}, {self.velocity[1]:.2f})\n"
            f"Wind: ({wind_at_pos[0]:.2f}, {wind_at_pos[1]:.2f})\n"
            f"Distance to goal: {np.linalg.norm(self.position - self.goal_position):.2f}\n"
            f"Action: {action_str}"
        )
        ax.text(0, -1, info_text, fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Create legend items
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2CA02C', 
                      label='Goal', markersize=10),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#D62728',
                      label='Boat', markersize=10),
            plt.Line2D([0], [0], color='white', marker='>', linestyle='',
                      label='Wind Direction', markersize=10),
        ]
        
        # Add velocity arrow to legend if it exists
        if velocity_arrow:
            legend_elements.append(
                plt.Line2D([0], [0], color='yellow', marker='>', linestyle='',
                          label='Boat Velocity', markersize=10)
            )
        
        # Add the legend
        ax.legend(handles=legend_elements, loc='upper left',
                 bbox_to_anchor=(0, 0), fontsize=9,
                 facecolor='white', framealpha=0.7)
        
        # Set title with better styling
        ax.set_title("Sailing Environment", fontsize=14, color='white', 
                     bbox=dict(facecolor=ocean_color, alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert plot to image
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        if self.render_mode == "human":
            # Display the image
            plt.imshow(img)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)
        
        return img
    
    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        # Map actions to direction vectors using the convention:
        # NORTH = (0, 1) (increasing Y)
        # SOUTH = (0, -1) (decreasing Y)
        # EAST = (1, 0) (increasing X)
        # WEST = (-1, 0) (decreasing X)
        directions = [
            (0, 1),     # 0: North (increasing Y)
            (1, 1),     # 1: Northeast
            (1, 0),     # 2: East (increasing X)
            (1, -1),    # 3: Southeast
            (0, -1),    # 4: South (decreasing Y)
            (-1, -1),   # 5: Southwest
            (-1, 0),    # 6: West (decreasing X)
            (-1, 1),    # 7: Northwest
            (0, 0)      # 8: Stay
        ]
        return np.array(directions[action])
    
    def _generate_wind_field(self):
        """Generate a new wind field with rotating anticyclonic patterns."""
        # Create a grid of coordinates
        x = np.linspace(0, self.grid_size[0] - 1, self.grid_size[0])
        y = np.linspace(0, self.grid_size[1] - 1, self.grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Get parameters
        p = self.wind_init_params
        base_x, base_y = p['base_direction']
        
        # Create the rotating pattern
        # Scale coordinates to match the pattern scale
        X_scaled = X / p['pattern_scale'] * 2 * np.pi
        Y_scaled = Y / p['pattern_scale'] * 2 * np.pi
        
        # Add random phase shift to make initial pattern random but smooth
        init_phase_shift = self.np_random.random() * 2 * np.pi
        
        # Create the rotating wind pattern
        # This creates a smooth rotation of the wind direction
        # The pattern rotates clockwise (anticyclonic in northern hemisphere)
        pattern_x = np.cos(X_scaled + Y_scaled + init_phase_shift)
        pattern_y = np.sin(X_scaled + Y_scaled + init_phase_shift)
        
        # Apply pattern strength to direction variations
        wind_x = base_x + p['pattern_strength'] * pattern_x
        wind_y = base_y + p['pattern_strength'] * pattern_y
        
        # Normalize the wind direction vectors
        magnitude = np.sqrt(wind_x**2 + wind_y**2)
        wind_x = wind_x / magnitude
        wind_y = wind_y / magnitude
        
        # Create strength variations following the same pattern but with different phase
        strength_phase_shift = self.np_random.random() * 2 * np.pi
        strength_pattern = 1.0 + p['strength_variation'] * (
            np.cos(X_scaled + Y_scaled + strength_phase_shift)
        )
        
        # Apply base speed and strength variations
        wind_x = wind_x * p['base_speed'] * strength_pattern
        wind_y = wind_y * p['base_speed'] * strength_pattern
        
        # Add noise component
        noise_x = p['noise'] * (self.np_random.random((self.grid_size[1], self.grid_size[0])) - 0.5)
        noise_y = p['noise'] * (self.np_random.random((self.grid_size[1], self.grid_size[0])) - 0.5)
        
        wind_x += noise_x
        wind_y += noise_y
        
        return np.stack([wind_x, wind_y], axis=-1)
    
    def _update_wind_field(self):
        """Update the wind field with small perturbations while maintaining spatial consistency."""
        # Create a grid of coordinates
        x = np.linspace(0, self.grid_size[0] - 1, self.grid_size[0])
        y = np.linspace(0, self.grid_size[1] - 1, self.grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Create smooth perturbation field using sine waves
        # This ensures spatial consistency in the changes
        perturbation_scale = self.wind_evol_params['pattern_scale'] / 4  # Smaller scale for perturbations
        X_scaled = X / perturbation_scale * 2 * np.pi
        Y_scaled = Y / perturbation_scale * 2 * np.pi
        
        # Add random phase shifts to make patterns different
        angle_phase_shift = self.np_random.random() * 2 * np.pi
        strength_phase_shift = self.np_random.random() * 2 * np.pi
        pattern_shift = np.pi / 3  # Fixed shift between angle and strength patterns
        
        # Generate smooth angle perturbations
        angle_amplitude = self.wind_evol_params['perturbation_angle_amplitude']
        perturbation_x = np.sin(X_scaled + Y_scaled + angle_phase_shift) * angle_amplitude
        perturbation_y = np.cos(X_scaled + Y_scaled + angle_phase_shift) * angle_amplitude
        
        # Add bias to angle perturbations
        bias_x, bias_y = self.wind_evol_params['wind_evolution_bias']
        bias_strength = self.wind_evol_params['bias_strength']
        if np.any(np.array([bias_x, bias_y]) != 0):  # Only apply if bias is non-zero
            # Normalize bias vector
            bias_norm = np.sqrt(bias_x**2 + bias_y**2)
            bias_x, bias_y = bias_x / bias_norm, bias_y / bias_norm
            # Add scaled bias to perturbations
            perturbation_x += bias_x * bias_strength * angle_amplitude
            perturbation_y += bias_y * bias_strength * angle_amplitude
        
        # Add angle perturbations to wind field
        self.wind_field[..., 0] += perturbation_x
        self.wind_field[..., 1] += perturbation_y
        
        # Normalize the wind vectors to maintain direction consistency
        magnitude = np.sqrt(np.sum(self.wind_field**2, axis=-1, keepdims=True))
        self.wind_field = self.wind_field / magnitude
        
        # Generate smooth strength perturbations (using a shifted pattern)
        strength_amplitude = self.wind_evol_params['perturbation_strength_amplitude']
        strength_perturbation = np.sin(X_scaled + Y_scaled + strength_phase_shift + pattern_shift)
        strength_factor = 1.0 + strength_perturbation * strength_amplitude
        
        # Apply strength perturbation while maintaining the base speed on average
        self.wind_field = self.wind_field * (self.wind_init_params['base_speed'] * strength_factor)[..., np.newaxis]
    
    def _get_wind_at_position(self, position):
        """Get wind vector at given position."""
        x, y = position
        # Numpy arrays are indexed with [y, x] order
        return self.wind_field[int(y), int(x)]
    
    def _calculate_new_velocity(self, current_velocity, wind, direction):
        """
        Calculate new velocity based on sailing physics.
        
        Args:
            current_velocity: Current velocity vector
            wind: Wind vector at current position
            direction: Desired direction vector (normalized)
            
        Returns:
            new_velocity: New velocity vector
        """
        # Calculate angle between wind and direction
        wind_norm = np.linalg.norm(wind)
        if wind_norm > 0:
            wind_normalized = wind / wind_norm
            direction_normalized = direction / np.linalg.norm(direction)
            
            # Calculate sailing efficiency using the shared function
            sailing_efficiency = calculate_sailing_efficiency(direction_normalized, wind_normalized)
            
            # Calculate theoretical velocity (what the boat would achieve with no inertia)
            theoretical_velocity = direction * sailing_efficiency * wind_norm * self.boat_performance
            
            # Apply max speed limit to theoretical velocity
            speed = np.linalg.norm(theoretical_velocity)
            if speed > self.max_speed:
                theoretical_velocity = (theoretical_velocity / speed) * self.max_speed
            
            # Apply inertia: new_velocity = theoretical_velocity + alpha*(old_velocity - theoretical_velocity)
            # where alpha is the inertia_factor
            new_velocity = theoretical_velocity + self.inertia_factor * (current_velocity - theoretical_velocity)
            
            # Ensure the new velocity doesn't exceed max speed
            speed = np.linalg.norm(new_velocity)
            if speed > self.max_speed:
                new_velocity = (new_velocity / speed) * self.max_speed
        else:
            # If no wind, just maintain some inertia
            new_velocity = self.inertia_factor * current_velocity
        
        return new_velocity
    
    def _calculate_reward(self, reached_goal, distance_to_goal):
        """
        Calculate reward based on current state.
        
        Args:
            reached_goal: Boolean indicating if the goal was reached
            distance_to_goal: Current distance to the goal
            
        Returns:
            reward: 100 if goal reached, 0 otherwise
        """
        if reached_goal:
            return 100.0
        return 0.0
    
    def _get_observation(self):
        """
        Create the observation array [x, y, vx, vy, wx, wy, flattened wind field].
        
        Returns:
            observation: A numpy array containing the agent's position, velocity,
                        the wind at the current position, and the full wind field.
        """
        # Get wind at current position
        current_wind = self._get_wind_at_position(self.position)
        
        # Flatten the wind field
        flattened_wind = self.wind_field.reshape(-1).astype(np.float32)
        
        # Create observation array
        observation = np.concatenate([
            self.position,      # x, y
            self.velocity,      # vx, vy
            current_wind,       # wx, wy
            flattened_wind      # Full wind field (flattened)
        ]).astype(np.float32)
        
        return observation 