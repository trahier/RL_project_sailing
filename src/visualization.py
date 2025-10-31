"""
Visualization utilities for sailing agents.

This module provides functions for visualizing agent trajectories and races.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from typing import Dict, Any, List, Optional
from ipywidgets import interact, IntSlider
from env_sailing import SailingEnv
from initial_windfields import get_initial_windfield
import io
from PIL import Image


def visualize_race(race_results: List[Dict[str, Any]], 
                   windfield_name: str, 
                   seed: int, 
                   max_steps: int,
                   show_full_trajectories: bool = False) -> None:
    """
    Visualize multiple agents racing on the same windfield with an interactive slider.
    
    Args:
        race_results: List of dictionaries containing race results for each agent
                     Each dict should have: name, color, positions, actions, reward, steps, success
        windfield_name: Name of the windfield to visualize
        seed: Seed used for the race
        max_steps: Maximum number of steps for the race
        show_full_trajectories: If True, show full trajectory for each agent (default: False)
    """
    # Create environment to get windfield
    initial_windfield = get_initial_windfield(windfield_name)
    env = SailingEnv(
        wind_init_params=initial_windfield['wind_init_params'],
        wind_evol_params=initial_windfield['wind_evol_params'],
        wind_grid_density=32,  # Show all 32x32 arrows
        wind_arrow_scale=120  # Higher scale = shorter arrows
    )
    env.seed(seed)
    env.reset(seed=seed)
    
    # Find the maximum number of steps among all agents
    max_race_steps = max(len(result['positions']) for result in race_results)
    
    def render_race_frame(step: int) -> None:
        """Render a single frame of the race at given step."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Set the limits
        ax.set_xlim(-1, env.grid_size[0] + 1)
        ax.set_ylim(-1, env.grid_size[1] + 1)
        
        # Ocean background
        ocean_color = '#6BAED6'
        ax.set_facecolor(ocean_color)
        ax.grid(True, linestyle='-', alpha=0.2, color='white')
        
        # Update environment to this step (for wind evolution)
        env.seed(seed)
        env.reset(seed=seed)
        for _ in range(step):
            env.step(0)  # Dummy step to evolve wind
        
        # Plot wind field
        step_size = max(1, env.grid_size[0] // env.wind_grid_density)
        x = np.arange(0, env.grid_size[0], step_size)
        y = np.arange(0, env.grid_size[1], step_size)
        X, Y = np.meshgrid(x, y)
        U = env.wind_field[::step_size, ::step_size, 0]
        V = env.wind_field[::step_size, ::step_size, 1]
        ax.quiver(X, Y, U, V, color='white', alpha=0.4, scale=120)
        
        # Draw goal
        goal_outer = Circle((env.goal_position[0], env.goal_position[1]), 1.5, 
                           color='green', alpha=0.3)
        goal_inner = Circle((env.goal_position[0], env.goal_position[1]), 1.0, 
                           color='#2CA02C', alpha=0.7)
        ax.add_patch(goal_outer)
        ax.add_patch(goal_inner)
        
        # Draw each agent
        legend_elements = []
        for result in race_results:
            # Check if agent has reached this step
            if step < len(result['positions']):
                position = result['positions'][step]
                
                # Calculate velocity direction (use difference from previous position if available)
                if step > 0:
                    prev_pos = result['positions'][step-1]
                    velocity = position - prev_pos
                else:
                    velocity = np.array([0, 1])  # Default north
                
                # Normalize velocity for direction
                if np.linalg.norm(velocity) > 0.1:
                    direction = velocity / np.linalg.norm(velocity)
                else:
                    direction = np.array([0, 1])
                
                # Draw boat as triangle
                boat_length = 1.5
                boat_width = 0.8
                
                bow = position + direction * boat_length * 0.6
                stern_center = position - direction * boat_length * 0.4
                port = stern_center + np.array([-direction[1], direction[0]]) * boat_width * 0.5
                starboard = stern_center + np.array([direction[1], -direction[0]]) * boat_width * 0.5
                
                boat_vertices = np.array([bow, port, stern_center, starboard])
                boat_polygon = Polygon(boat_vertices, color=result['color'], alpha=0.9, 
                                      edgecolor='black', linewidth=2)
                ax.add_patch(boat_polygon)
                
                # Draw trajectory trail
                if step > 0:
                    if show_full_trajectories:
                        # Show full trajectory up to current step
                        trail_positions = result['positions'][:step+1]
                        trail_x = [p[0] for p in trail_positions]
                        trail_y = [p[1] for p in trail_positions]
                        # Draw with gradient (older = more transparent)
                        for i in range(len(trail_x) - 1):
                            alpha = 0.15 + 0.25 * (i / len(trail_x))
                            ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                                   color=result['color'], alpha=alpha, linewidth=5)
                    else:
                        # Show only last 10 steps (original behavior)
                        trail_length = min(10, step)
                        trail_positions = result['positions'][max(0, step-trail_length):step+1]
                        trail_x = [p[0] for p in trail_positions]
                        trail_y = [p[1] for p in trail_positions]
                        ax.plot(trail_x, trail_y, color=result['color'], alpha=0.3, 
                               linewidth=5, linestyle='--')
                
                # Add to legend
                legend_elements.append(
                    plt.Line2D([0], [0], marker='^', color='w', 
                              markerfacecolor=result['color'], 
                              label=f"{result['name']}: Step {step}/{len(result['positions'])-1}",
                              markersize=12, markeredgecolor='black', markeredgewidth=2)
                )
            else:
                # Agent has finished - show at final position
                final_pos = result['positions'][-1]
                
                # Draw full trajectory for finished agent (if enabled)
                if show_full_trajectories and len(result['positions']) > 1:
                    trail_x = [p[0] for p in result['positions']]
                    trail_y = [p[1] for p in result['positions']]
                    # Draw with gradient (older = more transparent)
                    for i in range(len(trail_x) - 1):
                        alpha = 0.15 + 0.25 * (i / len(trail_x))
                        ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                               color=result['color'], alpha=alpha, linewidth=5)
                
                # Draw a marker at final position
                if result['success']:
                    ax.scatter(final_pos[0], final_pos[1], s=300, marker='*', 
                              color=result['color'], edgecolors='gold', linewidths=3,
                              zorder=10, label=f"{result['name']}: FINISHED!")
                else:
                    ax.scatter(final_pos[0], final_pos[1], s=200, marker='x', 
                              color=result['color'], linewidths=3,
                              zorder=10)
                
                legend_elements.append(
                    plt.Line2D([0], [0], marker='*' if result['success'] else 'x', 
                              color='w', markerfacecolor=result['color'],
                              label=f"{result['name']}: {'✅ FINISHED' if result['success'] else '❌ TIMEOUT'}",
                              markersize=12, markeredgecolor='gold' if result['success'] else 'black', 
                              markeredgewidth=2)
                )
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                 facecolor='white', framealpha=0.9)
        
        # Title
        ax.set_title(f"Race Visualization - Step {step}/{max_race_steps-1}\n" + 
                    f"Windfield: {windfield_name} | Seed: {seed}",
                    fontsize=14, color='white',
                    bbox=dict(facecolor=ocean_color, alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Add race info box
        info_lines = [f"🏁 Race Step: {step}"]
        for result in race_results:
            if step < len(result['positions']):
                pos = result['positions'][step]
                dist_to_goal = np.linalg.norm(pos - env.goal_position)
                info_lines.append(f"{result['name']}: {dist_to_goal:.1f} from goal")
            else:
                info_lines.append(f"{result['name']}: Finished at step {len(result['positions'])-1}")
        
        info_text = "\n".join(info_lines)
        ax.text(0, -1, info_text, fontsize=10, color='black',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
    
    # Create interactive slider
    interact(
        render_race_frame,
        step=IntSlider(
            min=0,
            max=max_race_steps-1,
            step=1,
            value=0,
            description='Race Step:'
        )
    )


def print_race_summary(race_results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary of race results.
    
    Args:
        race_results: List of dictionaries containing race results for each agent
    """
    import pandas as pd
    from IPython.display import display
    
    summary_data = []
    for result in race_results:
        summary_data.append({
            'Agent': result['name'],
            'Color': result['color'],
            'Steps': result['steps'],
            'Reward': f"{result['reward']:.2f}",
            'Success': '✅' if result['success'] else '❌'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Steps')  # Sort by steps (winner first)
    
    print("\n" + "="*50)
    print("🏆 RACE RESULTS 🏆")
    print("="*50)
    display(summary_df)
    
    # Determine winner
    successful_agents = [r for r in race_results if r['success']]
    if successful_agents:
        winner = min(successful_agents, key=lambda x: x['steps'])
        print(f"\n🥇 WINNER: {winner['name']} (completed in {winner['steps']} steps!)")
    else:
        print("\n❌ No agent reached the goal.")


def create_race_gif(race_results: List[Dict[str, Any]], 
                    windfield_name: str, 
                    seed: int, 
                    output_path: str,
                    fps: int = 10,
                    step_interval: int = 1,
                    figsize: tuple = (10, 10),
                    show_full_trajectories: bool = False) -> None:
    """
    Create a GIF animation of a race between multiple agents.
    
    Args:
        race_results: List of dictionaries containing race results for each agent
        windfield_name: Name of the windfield to visualize
        seed: Seed used for the race
        output_path: Path where the GIF will be saved (e.g., "race.gif")
        fps: Frames per second for the GIF (default: 10)
        step_interval: Interval between frames (1 = every step, 2 = every other step, etc.)
        figsize: Size of the figure (default: (10, 10))
        show_full_trajectories: If True, show full trajectory for each agent (default: False)
    
    Example:
        >>> create_race_gif(race_results, "static_headwind", 42, "my_race.gif", fps=15)
    """
    try:
        import imageio
    except ImportError:
        print("❌ Error: 'imageio' library is required to create GIFs.")
        print("Install it with: pip install imageio")
        return
    
    print(f"🎬 Creating race GIF...")
    
    # Create environment to get windfield
    initial_windfield = get_initial_windfield(windfield_name)
    env = SailingEnv(
        wind_init_params=initial_windfield['wind_init_params'],
        wind_evol_params=initial_windfield['wind_evol_params'],
        wind_grid_density=32,  # Show all 32x32 arrows
        wind_arrow_scale=120  # Higher scale = shorter arrows
    )
    env.seed(seed)
    env.reset(seed=seed)
    
    # Find the maximum number of steps among all agents
    max_race_steps = max(len(result['positions']) for result in race_results)
    
    # Generate frames
    frames = []
    steps_to_render = range(0, max_race_steps, step_interval)
    
    for step in steps_to_render:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set the limits
        ax.set_xlim(-1, env.grid_size[0] + 1)
        ax.set_ylim(-1, env.grid_size[1] + 1)
        
        # Ocean background
        ocean_color = '#6BAED6'
        ax.set_facecolor(ocean_color)
        ax.grid(True, linestyle='-', alpha=0.2, color='white')
        
        # Update environment to this step (for wind evolution)
        env.seed(seed)
        env.reset(seed=seed)
        for _ in range(step):
            env.step(0)  # Dummy step to evolve wind
        
        # Plot wind field
        step_size = max(1, env.grid_size[0] // env.wind_grid_density)
        x = np.arange(0, env.grid_size[0], step_size)
        y = np.arange(0, env.grid_size[1], step_size)
        X, Y = np.meshgrid(x, y)
        U = env.wind_field[::step_size, ::step_size, 0]
        V = env.wind_field[::step_size, ::step_size, 1]
        ax.quiver(X, Y, U, V, color='white', alpha=0.4, scale=120)
        
        # Draw goal
        goal_outer = Circle((env.goal_position[0], env.goal_position[1]), 1.5, 
                           color='green', alpha=0.3)
        goal_inner = Circle((env.goal_position[0], env.goal_position[1]), 1.0, 
                           color='#2CA02C', alpha=0.7)
        ax.add_patch(goal_outer)
        ax.add_patch(goal_inner)
        
        # Draw each agent
        legend_elements = []
        for result in race_results:
            if step < len(result['positions']):
                position = result['positions'][step]
                
                # Calculate velocity direction
                if step > 0:
                    prev_pos = result['positions'][step-1]
                    velocity = position - prev_pos
                else:
                    velocity = np.array([0, 1])
                
                if np.linalg.norm(velocity) > 0.1:
                    direction = velocity / np.linalg.norm(velocity)
                else:
                    direction = np.array([0, 1])
                
                # Draw boat
                boat_length = 1.5
                boat_width = 0.8
                
                bow = position + direction * boat_length * 0.6
                stern_center = position - direction * boat_length * 0.4
                port = stern_center + np.array([-direction[1], direction[0]]) * boat_width * 0.5
                starboard = stern_center + np.array([direction[1], -direction[0]]) * boat_width * 0.5
                
                boat_vertices = np.array([bow, port, stern_center, starboard])
                boat_polygon = Polygon(boat_vertices, color=result['color'], alpha=0.9, 
                                      edgecolor='black', linewidth=2)
                ax.add_patch(boat_polygon)
                
                # Draw trail
                if step > 0:
                    if show_full_trajectories:
                        # Show full trajectory up to current step
                        trail_positions = result['positions'][:step+1]
                        trail_x = [p[0] for p in trail_positions]
                        trail_y = [p[1] for p in trail_positions]
                        # Draw with gradient (older = more transparent)
                        for i in range(len(trail_x) - 1):
                            alpha = 0.15 + 0.25 * (i / len(trail_x))
                            ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                                   color=result['color'], alpha=alpha, linewidth=5)
                    else:
                        # Show only last 10 steps (original behavior)
                        trail_length = min(10, step)
                        trail_positions = result['positions'][max(0, step-trail_length):step+1]
                        trail_x = [p[0] for p in trail_positions]
                        trail_y = [p[1] for p in trail_positions]
                        ax.plot(trail_x, trail_y, color=result['color'], alpha=0.3, 
                               linewidth=5, linestyle='--')
                
                legend_elements.append(
                    plt.Line2D([0], [0], marker='^', color='w', 
                              markerfacecolor=result['color'], 
                              label=f"{result['name']}",
                              markersize=10, markeredgecolor='black', markeredgewidth=1.5)
                )
            else:
                final_pos = result['positions'][-1]
                
                # Draw full trajectory for finished agent (if enabled)
                if show_full_trajectories and len(result['positions']) > 1:
                    trail_x = [p[0] for p in result['positions']]
                    trail_y = [p[1] for p in result['positions']]
                    # Draw with gradient (older = more transparent)
                    for i in range(len(trail_x) - 1):
                        alpha = 0.15 + 0.25 * (i / len(trail_x))
                        ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                               color=result['color'], alpha=alpha, linewidth=5)
                
                if result['success']:
                    ax.scatter(final_pos[0], final_pos[1], s=250, marker='*', 
                              color=result['color'], edgecolors='gold', linewidths=2.5,
                              zorder=10)
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='*', color='w', 
                                  markerfacecolor=result['color'],
                                  label=f"{result['name']} ✅",
                                  markersize=10, markeredgecolor='gold', markeredgewidth=1.5)
                    )
                else:
                    ax.scatter(final_pos[0], final_pos[1], s=150, marker='x', 
                              color=result['color'], linewidths=2.5, zorder=10)
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='x', color='w', 
                                  markerfacecolor=result['color'],
                                  label=f"{result['name']} ❌",
                                  markersize=10, markeredgecolor='black', markeredgewidth=1.5)
                    )
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                 facecolor='white', framealpha=0.9)
        
        # Title
        ax.set_title(f"Step {step}/{max_race_steps-1} | {windfield_name}",
                    fontsize=12, color='white', pad=10,
                    bbox=dict(facecolor=ocean_color, alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        
        plt.close(fig)
        buf.close()
        
        # Progress indicator
        if (len(frames) % 10 == 0) or (step == max_race_steps - 1):
            print(f"  Processed {len(frames)}/{len(steps_to_render)} frames...", end='\r')
    
    print(f"\n💾 Saving GIF to {output_path}...")
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    
    print(f"✅ GIF created successfully! Saved to: {output_path}")
    print(f"   Total frames: {len(frames)} | Duration: ~{len(frames)/fps:.1f} seconds")

