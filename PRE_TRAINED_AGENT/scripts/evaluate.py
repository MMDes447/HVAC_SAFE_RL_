import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch as T

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PPOAgent1_epch
from src.environment import create_environment

def evaluate(n_eval_episodes=3):
    # Setup BOPTEST connection
    url = 'https://api.boptest.net'
    testcase = 'bestest_hydronic_heat_pump'
    
    # Select and initialize test case
    import requests
    testid = requests.post(f'{url}/testcases/{testcase}/select').json()['testid']
    
    # Create environment
    env = create_environment(url, testcase)
    
    # Initialize agent
    agent = PPOAgent1_epch(
        n_actions=env.action_space.n,
        batch_size=6,
        alpha=0.001,
        n_epochs=10,
        input_dims=1
    )
    
    # Load trained models
    agent.load_models()
    
    # Tracking metrics
    score_history = []
    safe_episodes = 0
    unsafe_episodes = 0
    
    print("Starting evaluation...")
    for i in range(n_eval_episodes):
        observation, _ = env.reset()
        done = False
        score = 0
        episode_unsafe = False
        
        while not done:
            # Select action
            action, _, _ = agent.choose_action(observation)
            
            # Take step in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            
            # Check if state is unsafe
            if agent.is_unsafe(np.array(observation)):
                episode_unsafe = True
                
            observation = next_observation
            
        # Update episode safety counters
        if episode_unsafe:
            unsafe_episodes += 1
        else:
            safe_episodes += 1
            
        score_history.append(score)
        avg_score = np.mean(score_history)
        
        print(f'Episode {i+1}')
        print(f'Score: {score:.2f}, Avg Score: {avg_score:.2f}')
        print(f'Safe Episodes: {safe_episodes}, Unsafe Episodes: {unsafe_episodes}')
        print(f'Safety Ratio: {(safe_episodes/(i+1))*100:.2f}%\n')
        
    print(f'Evaluation completed.')
    print(f'Final average score: {avg_score:.2f}')
    print(f'Total Safe Episodes: {safe_episodes}, Total Unsafe Episodes: {unsafe_episodes}')
    print(f'Final Safety Ratio: {(safe_episodes/n_eval_episodes)*100:.2f}%')
    
    # Plot barrier function
    plot_barrier_function(agent)
    
    env.close()

def plot_barrier_function(agent):
    """Plot the learned barrier function across temperature range"""
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate temperatures in Kelvin
    min_temp = 5 + 273.15
    max_temp = 29 + 273.15
    n_samples = 10000
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
        
    # Create the plot
    plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Barrier values vs. Kelvin temperatures
    ax1.scatter(temps_K, barrier_values, alpha=0.5, c='blue', label='Barrier Values')
    ax1.axvline(x=lower_bound, color='r', linestyle='--', label='Safety Bounds')
    ax1.axvline(x=upper_bound, color='r', linestyle='--')
    ax1.axvspan(lower_bound, upper_bound, alpha=0.2, color='green', label='Safe Region')
    ax1.axvspan(min_temp, lower_bound, alpha=0.2, color='red', label='Unsafe Region')
    ax1.axvspan(upper_bound, max_temp, alpha=0.2, color='red')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Barrier Function Value')
    ax1.set_title('Barrier Function vs. Absolute Temperature')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Barrier values vs. Celsius temperatures
    temps_C = temps_K - 273.15
    ax2.scatter(temps_C, barrier_values, alpha=0.5, c='blue', label='Barrier Values')
    ax2.axvline(x=18, color='r', linestyle='--', label='Safety Bounds')
    ax2.axvline(x=21, color='r', linestyle='--')
    ax2.axvspan(18, 21, alpha=0.2, color='green', label='Safe Region')
    ax2.axvspan(min_temp-273.15, 18, alpha=0.2, color='red', label='Unsafe Region')
    ax2.axvspan(21, max_temp-273.15, alpha=0.2, color='red')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Barrier Function Value')
    ax2.set_title('Barrier Function vs. Celsius Temperature')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/barrier_function.png')
    plt.show()

if __name__ == "__main__":
    evaluate()