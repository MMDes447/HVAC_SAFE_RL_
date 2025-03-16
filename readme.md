# HVAC Safety Barrier RL

This repository contains code for the paper "Safe Reinforcement Learning for Buildings: Minimizing Energy Use While Maximizing Occupant Comfort." It implements reinforcement learning with neural barrier certificates for HVAC control systems using BOPTEST as the training environment. The experiments examine:

- Safe HVAC control policies that balance energy efficiency and comfort
- Effects of barrier certificate pre-training versus no pre-training
- Performance with random versus non-random episode generation
- Impact of different episode lengths on training outcomes



[an_awesome_website_link]: https://ibpsa.github.io/project1-boptest/ 


## Project Structure
<pre>

HVAC-SAFETY-BARRIER-RL

├── NOTEBOOKS
│ ├── Occupancy_aware_Agent.ipynb
│ └── PRE_TRAINED_AND_STANDARD(...)
├── OCCUPANCY_AWARE_AGENT
│ ├── logs
│ ├── scripts
│ │ ├── evaluate.py
│ │ ├── train.py
│ │ └── visualize.py
│ ├── src
│ │ ├── agent.py
│ │ ├── environment.py
│ │ ├── networks.py
│ │ └── memory.py
├── PRE_TRAINED_AGENT
│ ├── logs
│ ├── scripts
│ │ ├── evaluate.py
│ │ ├── train.py
│ │ └── visualize.py
│ ├── src
│ │ ├── agent.py
│ │ ├── environment.py
│ │ ├── networks.py
│ │ └── memory.py
├── Standard Agent
│ ├── logs
│ ├── scripts
│ │ ├── evaluate.py
│ │ ├── train.py
│ │ └── visualize.py
│ ├── src
│ │ ├── agent.py
│ │ ├── environment.py
│ │ ├── networks.py
│ │ └── memory.py
├── readme.md
└── requirements.txt
</pre>
## Agents

### Occupancy-Aware Agent

Occupancy-aware agent uses occupancy data to train a safe policy.

### Pre-Trained Agent

First pre-training the barrier and then the joint learning of barrier certificate and agent. 
### Standard Agent
joint learning of agent and barrie without pre-training

## Notebooks

The `NOTEBOOKS/` directory contains Jupyter notebooks for interactive experimentation:
- `Occupancy_aware_Agent.ipynb`
- `PRE_TRAINED_AND_STANDARD(1).ipynb`
## Install BOPTESTGYM
```
git clone -b boptest-gym-service https://github.com/ibpsa/project1-boptest-gym.git boptestGymService
```
## To run the experiments 

```bash
python {Agent_directory}/scripts/train.py --episodes {specify the number} --length {specify the number} --step_period {specify the number} --barrier_only {specify the number}# Clone the repository
To disable the random episode generation
# To disable random episodes
python train.py --random False

