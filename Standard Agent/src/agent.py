import numpy as np
import torch as T
import torch.optim as optim

from src.networks import ActorNetwork1, CriticNetwork1, BarrierNet
from src.memory import PPOMemory1

class PPOAgent1_epch:
    def __init__(self, n_actions, input_dims, alpha=0.0003, batch_size=64,
                 n_epochs=10, gae_lambda=0.95, gamma=0.99, policy_clip=0.2):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.global_step = 0
        self.epsilon = .01

        # PPO networks
        self.actor = ActorNetwork1(n_actions, input_dims, alpha)
        self.critic = CriticNetwork1(input_dims, alpha)
        self.memory = PPOMemory1(batch_size)

        # Barrier components
        self.barrier = BarrierNet(input_dims)
        self.barrier_optimizer = optim.Adam(self.barrier.parameters(), lr=0.001)
        self.lambda_param = 0.01
        self.barrier_memory = {
            'initial_states': [],
            'unsafe_states': [],
            'states': [],
            'next_states': []
        }

        # Lagrangian optimization parameters
        self.nu = T.tensor(1.00, requires_grad=True)
        self.alpha_dual = .01  # Learning rate for dual ascent
        self.nu_optimizer = optim.Adam([self.nu], lr=self.alpha_dual)
        
        # Temperature bounds
        self.lower_temp = 18 + 273.15
        self.upper_temp = 21 + 273.15

    def is_safe(self, state):
        return (-1 <= state[0] <= 1)

    def is_unsafe(self, state):
        return (-1 > state[0] or state[0] > 1)

    def remember(self, state, action, probs, vals, reward, done, next_state):
        # PPO memory
        self.memory.store_memory(state, action, probs, vals, reward, done)

        # Barrier memory
        state_arr = np.array(state)
        next_state_arr = np.array(next_state)

        self.barrier_memory['states'].append(state_arr)

        if self.is_unsafe(state_arr):
            self.barrier_memory['unsafe_states'].append(state_arr)
        if self.is_safe(state_arr):
            self.barrier_memory['initial_states'].append(state_arr)

        if not done:  # Only add next_state if episode isn't done
            self.barrier_memory['next_states'].append(next_state_arr)

    def barrier_feasible_loss(self, initial_states):
        barrier_vals = self.barrier(initial_states)
        return T.mean(T.max(barrier_vals, T.zeros_like(barrier_vals)))

    def barrier_infeasible_loss(self, unsafe_states):
        barrier_vals = self.barrier(unsafe_states)
        return T.mean(T.max(-barrier_vals, T.zeros_like(barrier_vals)))

    def barrier_invariant_loss(self, states, next_states):
        current_barrier = self.barrier(states)
        next_barrier = self.barrier(next_states)
        return T.max(next_barrier - (1 - self.lambda_param) * current_barrier,
                 T.zeros_like(current_barrier))

    def learn_barrier(self):
        if (len(self.barrier_memory['initial_states']) > 0 and
            len(self.barrier_memory['unsafe_states']) > 0 and
            len(self.barrier_memory['next_states']) > 0):
            
            bar_epoch = 30
            num_transitions = len(self.barrier_memory['next_states'])
            
            for _ in range(bar_epoch):
                initial_states = T.tensor(np.array(self.barrier_memory['initial_states']),
                                      dtype=T.float).to(self.barrier.device)
                unsafe_states = T.tensor(np.array(self.barrier_memory['unsafe_states']),
                                      dtype=T.float).to(self.barrier.device)
                states = T.tensor(np.array(self.barrier_memory['states'][:num_transitions]),
                                dtype=T.float).to(self.barrier.device)
                next_states = T.tensor(np.array(self.barrier_memory['next_states']),
                                    dtype=T.float).to(self.barrier.device)

                self.barrier_optimizer.zero_grad()
                barrier_loss = (self.barrier_feasible_loss(initial_states) +
                              self.barrier_infeasible_loss(unsafe_states) +
                              self.barrier_invariant_loss(states, next_states).mean())
                barrier_loss.backward()
                self.barrier_optimizer.step()
                print(f'Barrier loss: {barrier_loss.item()}')

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, done_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # GAE calculation
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*
                    (1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # PPO update with Lagrangian constraint
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # Get next states for the batch
                next_indices = [min(i + 1, len(state_arr) - 1) for i in batch]
                next_states = T.tensor(state_arr[next_indices], dtype=T.float).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)

                # Calculate importance sampling ratio
                prob_ratio = new_probs.exp()/old_probs.exp()

                # PPO actor loss components
                weighted_probs = advantage[batch]*prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                1+self.policy_clip)*advantage[batch]

                # Original PPO actor loss
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate invariant loss with importance sampling
                J_invt = self.barrier_invariant_loss(states, next_states).detach()
                importance_weighted_J_invt = (prob_ratio * J_invt).mean()

                constrained_actor_loss = actor_loss + (importance_weighted_J_invt)*(self.nu.detach())
                
                # Critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                constrained_actor_loss.backward()
                critic_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                # Nu update
                self.nu_optimizer.zero_grad()
                loss_nu = -self.nu * (importance_weighted_J_invt.detach())
                loss_nu.backward()
                self.nu_optimizer.step()
                T.clamp(self.nu, min=0.0)
                
                self.global_step += 1
                
                # Log training metrics
                if hasattr(self, 'writer'):
                    self.log_training_metrics(
                        self.writer,  
                        self.global_step,
                        actor_loss,
                        critic_loss,
                        importance_weighted_J_invt,
                        constrained_actor_loss)

        self.memory.clear_memory()

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
        # Save barrier network
        T.save(self.barrier.state_dict(), 'models/barrier_net.pt')
        
        # Save Lagrange multiplier
        T.save(self.nu, 'models/lagrange_multiplier.pt')

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
        # Load barrier network if file exists
        if os.path.exists('models/barrier_net.pt'):
            self.barrier.load_state_dict(T.load('models/barrier_net.pt'))
            
        # Load Lagrange multiplier if file exists
        if os.path.exists('models/lagrange_multiplier.pt'):
            self.nu = T.load('models/lagrange_multiplier.pt')

    def log_training_metrics(self, writer, step,
                        actor_loss, critic_loss,
                        importance_weighted_J_invt, constrained_actor_loss):
        # Log training metrics to TensorBoard
        writer.add_scalar('Losses/Actor_Loss', actor_loss.item(), step)
        writer.add_scalar('Losses/Critic_Loss', critic_loss.item(), step)
        writer.add_scalar('Losses/Constrained_Actor_Loss', constrained_actor_loss.item(), step)

        # Lagrangian metrics
        writer.add_scalar('Lagrangian/Nu_Value', self.nu.item(), step)
        writer.add_scalar('Lagrangian/Importance_Weighted_Constraint',
                      importance_weighted_J_invt.item(), step)

    def evaluate_barrier(self, state):
        with T.no_grad():
            state_tensor = T.tensor([state], dtype=T.float).to(self.barrier.device)
            return self.barrier(state_tensor).item()