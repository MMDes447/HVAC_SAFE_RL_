# HVAC Safety Barrier RL

A reinforcement learning approach for HVAC systems with safety constraints and occupancy awareness.

## Project Structure
HVAC-SAFETY-BARRIER-RL/
├── NOTEBOOKS/
│ ├── Occupancy_aware_Agent.ipynb
│ └── PRE_TRAINED_AND_STANDARD(...)
├── OCCUPANCY_AWARE_AGENT/
│ ├── logs/
│ ├── scripts/
│ │ ├── evaluate.py
│ │ ├── train.py
│ │ └── visualize.py
│ ├── src/
│ ├── readme.md
│ └── requirement.txt
├── PRE_TRAINED_AGENT/
│ ├── logs/
│ ├── scripts/
│ │ ├── evaluate.py
│ │ ├── train.py
│ │ └── visualize.py
│ ├── src/
│ ├── readme.md
│ └── requirement.txt
└── Standard Agent/
├── logs/
├── scripts/
├── src/
├── readme.md
└── requirement.txt
## Overview

This repository implements three different reinforcement learning agents for HVAC system control with a focus on energy efficiency, occupant comfort, and safety constraints. The project provides a comprehensive framework for developing and evaluating advanced control strategies for building management systems.

## Agents

### Occupancy-Aware Agent

Located in `OCCUPANCY_AWARE_AGENT/`, this implementation adapts HVAC control strategies based on real-time and predicted occupancy patterns. The agent optimizes comfort when spaces are occupied while reducing energy consumption during unoccupied periods.

- **Training**: Run `scripts/train.py` to train the agent
- **Evaluation**: Use `scripts/evaluate.py` to test performance
- **Visualization**: Generate performance graphs with `scripts/visualize.py`

### Pre-Trained Agent

Located in `PRE_TRAINED_AGENT/`, this provides ready-to-use models for immediate deployment. These models have been trained with safety constraints to ensure reliable operation across a range of building environments.

- **Evaluation**: Use `scripts/evaluate.py` to test on your specific environment
- **Visualization**: Generate performance reports with `scripts/visualize.py`

### Standard Agent

Located in `Standard Agent/`, this implements baseline control strategies for comparative analysis. It follows conventional HVAC control approaches to provide a benchmark for evaluating the advanced RL-based agents.

## Notebooks

The `NOTEBOOKS/` directory contains Jupyter notebooks for interactive experimentation and analysis:

- `Occupancy_aware_Agent.ipynb`: Demonstrates the occupancy-aware agent's capabilities
- `PRE_TRAINED_AND_STANDARD(...)`: Compares pre-trained and standard agents

## Installation

```bash
# Clone the repository
git clone https://github.com/MMDes447/HVAC-SAFETY-BARRIER-RL.git

# Navigate to project directory
cd HVAC-SAFETY-BARRIER-RL

# Install common dependencies
pip install -r requirement.txt
