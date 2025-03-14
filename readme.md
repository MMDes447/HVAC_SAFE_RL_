# HVAC Safety Barrier RL
This repositoty contains the code for the experiments in the paper "Safe Reinforcement Learning for Buildings:
Minimizing Energy Use While Maximizing Occupant Comfort".  JAFAR\


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

│ ├── readme.md

│ └── requirement.txt

├── PRE_TRAINED_AGENT

│ ├── logs

│ ├── scripts

│ │ ├── evaluate.py

│ │ ├── train.py

│ │ └── visualize.py

│ ├── src

│ ├── readme.md

│ └── requirement.txt

├── Standard Agent

│ ├── logs

│ ├── scripts

│ ├── src

│ ├── readme.md

│ └── requirement.txt

├── readme.md

└── requirement.txt

</pre>
## Overview

This repository implements three different reinforcement learning agents for HVAC system control with a focus on energy efficiency, occupant comfort, and safety constraints.
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

