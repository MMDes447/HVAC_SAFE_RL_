import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorNetwork1(nn.Module):
  def __init__(self,n_actions,input_dims,alpha,fc1_dims=256,fc2_dims=256,chkpt_dir='tmp/ppo'):
    super(ActorNetwork1, self).__init__()
    self.checkpoint_file=os.path.join(chkpt_dir,'actor_ppo')
    self.actor=nn.Sequential(
        nn.Linear(input_dims,fc1_dims),
        nn.ReLU(),
        nn.Linear(fc1_dims,fc2_dims),
        nn.ReLU(),
        nn.Linear(fc2_dims,n_actions),
        nn.Softmax(dim=-1)
    )
    self.optimizer=optim.Adam(self.parameters(),lr=alpha)
    self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)
  def forward(self, state):
    dist=self.actor(state)
    dist=Categorical(dist)
    return dist
  def save_checkpoint(self):
    T.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork1(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork1, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class BarrierNet(nn.Module):
  def __init__(self, input_dims):
    super(BarrierNet, self).__init__()
    self.netwrok=nn.Sequential(nn.Linear(input_dims,256),
                               nn.Tanh(),
                               nn.Linear(256,256),
                               nn.Tanh(),
                               nn.Linear(256,1)
                               )
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)
  def forward(self,state):
    return self.netwrok(state)
