import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch as T

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PPOAgent1_epch

def visualize_barrier_function(model_path='models', save_path='plots'):
    """Visualize the learned barrier function"""
    
    # Create directory for plots
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize agent just for the barrier function
    agent = PPOAgent1_epch(n_actions=12, input_dims=1)
    
    # Load model
    agent.load_models()
    
    # Generate temperatures in Kelvin
    min_temp = 5 + 273.15
    max_temp = 29 + 273.15
    n_samples = 1000
    temps_K = np.linspace(min_temp, max_temp, n_samples)
    
    # Normalize temperatures for barrier evaluation
    lower_bound = 18 + 273.15
    upper_bound = 21 + 273.15
    temps_normalized = 2 * (temps_K - lower_bound) / (upper_bound - lower_bound) - 1
    
    # Evaluate barrier function for each normalized temperature
    barrier_values = []
    for temp in temps_normalized:
        barrier_val = agent.evaluate_barrier(temp)
        barrier_values.append(barrier_val)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(temps_K - 273.15, barrier_values, alpha=0.5, c='blue', label='Barrier Values')
    plt.axvline(x=18, color='r', linestyle='--', label='Safety Bounds (18°C)')
    plt.axvline(x=21, color='r', linestyle='--', label='Safety Bounds (21°C)')
    plt.axvspan(18, 21, alpha=0.2, color='green', label='Safe Region')
    plt.axvspan(min_temp-273.15, 18, alpha=0.2, color='red', label='Unsafe Region')
    plt.axvspan(21, max_temp-273.15, alpha=0.2, color='red')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Barrier Function Value')
    plt.title('Safety Barrier Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(save_path, 'barrier_function.png'))
    plt.show()

if __name__ == "__main__":
    visualize_barrier_function()