import requests
import numpy as np
import gymnasium as gym
import sys
sys.path.insert(0, 'boptestGymService')
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper

class BoptestGymEnvCustomReward(BoptestGymEnv):
    '''Define a custom reward for this building'''
    
    def get_reward(self):
        '''Custom reward function that penalizes energy and cost'''
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        objective_integrand = 10*kpis['ener_tot'] + 10*kpis['cost_tot']
        # Calculate reward as negative change in objective
        reward = -(objective_integrand - self.objective_integrand)
        self.objective_integrand = objective_integrand
        return reward

def create_environment(url, testcase='bestest_hydronic_heat_pump', RANDOM_EPISODES=True, LEN=2*24, step_period=1800 ):
    """Create a custom BOPTEST environment with appropriate wrappers"""
    
    # Temperature setpoints for normalizing
    lower_setp = 18 + 273.15
    upper_setp = 21 + 273.15
    
    
    # Instantiate environment
    env = BoptestGymEnvCustomReward(
        url=url,
        testcase=testcase,
        actions=['oveHeaPumY_u'],
        observations={'reaTZon_y': (lower_setp, upper_setp),'Occupancy[0]':(0,5)},
        random_start_time=RANDOM_EPISODES,
        max_episode_length=LEN*3600,
        warmup_period=24*3600,
        step_period=step_period,
        render_episodes=True
    )
    
    # Add observation normalization
    env = NormalizedObservationWrapper(env)
    
    # Add action discretization
    env = DiscretizedActionWrapper(env, n_bins_act=12)
    
    return env