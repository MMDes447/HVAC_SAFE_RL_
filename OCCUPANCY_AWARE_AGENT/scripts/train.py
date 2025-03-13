import os
import sys
import random
import numpy as np
import torch as T
import argparse
from torch.utils.tensorboard import SummaryWriter

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PPOAgent1_epch
from src.environment import create_environment

def train(n_episodes=20, random_episodes=True, episode_length=48, step_period=1800):
    # Set up logging directory
    os.makedirs('logs', exist_ok=True)
    writer = SummaryWriter('logs/safety_ppo_experiment')

    # Set seeds for reproducibility
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)

    # Setup BOPTEST connection
    url = 'https://api.boptest.net'
    testcase = 'bestest_hydronic_heat_pump'
    
    # Select and initialize test case
    import requests
    testid = requests.post(f'{url}/testcases/{testcase}/select').json()['testid']
    
    # Create environment
    env = create_environment(url, testcase, RANDOM_EPISODES=random_episodes, 
                            LEN=episode_length, step_period=step_period)
    
    # Initialize agent
    agent = PPOAgent1_epch(
        n_actions=env.action_space.n,
        batch_size=6,
        alpha=0.001,
        n_epochs=10,
        input_dims=2
    )
    agent.writer = writer  # Set writer for logging
    
    # Training parameters
    learning_freq = 48  # How often to perform learning updates
    barrier_only_episodes = 20  # Number of episodes with only barrier learning
    
    # Initialize tracking variables
    n_steps = 0
    learn_iters = 0
    score_history = []
    best_score = float('-inf')
    
    print("Starting training...")
    print(f"Configuration: Episodes={n_episodes}, Random Episodes={random_episodes}, " 
          f"Episode Length={episode_length}, Step Period={step_period}s")
    print(f"First {barrier_only_episodes} episodes: Only barrier learning")
    print(f"After episode {barrier_only_episodes}: Both barrier and policy learning")
    
    for i in range(n_episodes):
        observation, _ = env.reset()
        done = False
        score = 0
        episode_steps = 0
        
        # For tracking episode-specific safety metrics
        episode_unsafe_visits = 0
        episode_safe_visits = 0
        
        while not done:
            # Select action
            action, prob, val = agent.choose_action(observation)
            
            # Take step in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update counters
            n_steps += 1
            episode_steps += 1
            score += reward
            
            # Track safety violations
            if agent.is_unsafe(np.array(observation)):
                episode_unsafe_visits += 1
            if agent.is_safe(np.array(observation)):
                episode_safe_visits += 1
                
            # Store transition
            agent.remember(observation, action, prob, val, reward, done, next_observation)
            
            # Periodic learning
            if n_steps % learning_freq == 0:
                # Log current state
                writer.add_scalar('Barrier/States/Safe_Count',
                                len(agent.barrier_memory['initial_states']), n_steps)
                writer.add_scalar('Barrier/States/Unsafe_Count',
                                len(agent.barrier_memory['unsafe_states']), n_steps)
                writer.add_scalar('Barrier/States/Total_Transitions',
                                len(agent.barrier_memory['next_states']), n_steps)
                
                # Learning - first 20 episodes only barrier learning, after that both
                agent.learn_barrier()  # Always learn barrier function
                
                if i >= barrier_only_episodes:  # After first 20 episodes, also learn policy
                    agent.learn()
                
                learn_iters += 1
                
            observation = next_observation
            
        # Episode complete - log episode-level metrics
        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else score
        
        # Performance metrics
        writer.add_scalar('Performance/Episode_Score', score, i)
        writer.add_scalar('Performance/Average_Score', avg_score, i)
        writer.add_scalar('Performance/Episode_Length', episode_steps, i)
        
        # Safety metrics
        writer.add_scalar('Safety/Unsafe_Visits_Per_Episode', episode_unsafe_visits, i)
        writer.add_scalar('Safety/Safe_Visits_Per_Episode', episode_safe_visits, i)
        safety_ratio = episode_safe_visits / max(1, (episode_safe_visits + episode_unsafe_visits))
        writer.add_scalar('Safety/Safety_Ratio', safety_ratio, i)
        
        # Learning metrics
        writer.add_scalar('Training/Learning_Steps', learn_iters, i)
        writer.add_scalar('Training/Total_Steps', n_steps, i)
        
        print(f'Episode {i+1}')
        print(f'Score: {score:.2f}, Avg Score: {avg_score:.2f}')
        print(f'Time Steps: {n_steps}, Learning Steps: {learn_iters}')
        print(f'Safe states: {len(agent.barrier_memory["initial_states"])}')
        print(f'Unsafe states: {len(agent.barrier_memory["unsafe_states"])}')
        print(f'Total transitions: {len(agent.barrier_memory["next_states"])}')
        
        # Current learning phase
        if i < barrier_only_episodes:
            print("Current phase: BARRIER LEARNING ONLY")
        else:
            print("Current phase: BARRIER AND POLICY LEARNING")
        
        # Save if best model
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            
    print(f'Training completed. Final average score: {avg_score:.2f}')
    writer.close()
    env.close()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a PPO agent with safety constraints')
    parser.add_argument('--episodes', type=int, default=40, 
                        help='Number of episodes to train (default: 40)')
    parser.add_argument('--random', type=bool, default=True,
                        help='Whether to use random episodes (default: True)')
    parser.add_argument('--length', type=int, default=48,
                        help='Episode length in steps (default: 48, equivalent to 2*24)')
    parser.add_argument('--step_period', type=int, default=1800,
                        help='Step period in seconds (default: 1800)')
    parser.add_argument('--barrier_only', type=int, default=20,
                        help='Number of episodes with only barrier learning (default: 20)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run training with the specified parameters
    train(n_episodes=args.episodes, 
          random_episodes=args.random, 
          episode_length=args.length, 
          step_period=args.step_period)