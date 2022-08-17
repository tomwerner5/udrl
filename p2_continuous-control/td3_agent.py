import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 5e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate
LR_CRITIC_1 = 1e-3  # learning rate
LR_CRITIC_2 = 1e-3  # learning rate
UPDATE_EVERY = 1  # how often the agent should learn
POLICY_UPDATE_FREQUENCY = 2 # How often to update the policy
SEED = 1  # random state

np.random.seed(SEED)
random.seed(SEED)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, actor, critic, device='cpu'):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            actor (object): actor network in TD3, a pytorch object of type torch.nn.Module or similar
            critic (object): critic network in TD3, a pytorch object of type torch.nn.Module or similar
            device (str): device for value placement during training, either gpu or cpu (e.g. `cuda:0`)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # TD3 local and target networks creation
        self.actor_local = actor(state_size, action_size, SEED).to(device)
        self.actor_target = actor(state_size, action_size, SEED).to(device)
        self.critic_1_local = critic(state_size, action_size, SEED).to(device)
        self.critic_1_target = critic(state_size, action_size, SEED).to(device)
        self.critic_2_local = critic(state_size, action_size, SEED+1).to(device)
        self.critic_2_target = critic(state_size, action_size, SEED+1).to(device)

        # network optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_1_optimizer = optim.Adam(self.critic_1_local.parameters(), lr=LR_CRITIC_1)
        self.critic_2_optimizer = optim.Adam(self.critic_2_local.parameters(), lr=LR_CRITIC_2)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Initialize noise
        self.policy_noise = GaussianNoise(action_size, sigma=0.2, clip_value=0.5)
        self.action_noise = GaussianNoise(action_size, sigma=0.1)

    def step(self, state, action, reward, next_state, done, t):
        """Take a step in the environment.

        Params
        ======
            state (array-like): current state
            action (array-like): current action
            reward (float): current reward
            next_state (array-like): immediate next state
            done (bool): boolean for terminal state
            t (int): current time step
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, t)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            add_noise (bool): adds OU noise to the actions
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            # calculate actions using policy network
            actions = self.actor_local(state)
        self.actor_local.train()
        if add_noise:
            # add noise to the action selection
            actions = actions + self.action_noise(False)

        # clip the action to the action space min/max
        return actions.clamp(-1, 1)

    def learn(self, experiences, gamma, t):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            t (int): current time step
        """
        # sampled experiences from buffer
        states, actions, rewards, next_states, dones = experiences

        # Calculate next actions using the clipped noise
        next_actions = self.actor_target(next_states) + self.policy_noise(True)

        # calculate action-values from each critic network using the calculated next actions
        q_target_1 = self.critic_1_target(next_states,
                                          next_actions)
        q_target_2 = self.critic_2_target(next_states,
                                          next_actions)

        # calculate the overall action value of the targets by using the minimum of the two critic values
        q_target_value = rewards + (gamma * torch.min(q_target_1, q_target_2) * (1 - dones))

        # calculate the action value of the local networks using sampled values
        q_local_value_1 = self.critic_1_local(states,
                                              actions)
        q_local_value_2 = self.critic_2_local(states,
                                              actions)

        # critic 1 update
        critic_1_loss = F.mse_loss(q_target_value, q_local_value_1)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()

        # critic 2 update
        critic_2_loss = F.mse_loss(q_target_value, q_local_value_2)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # actor update
        if t % POLICY_UPDATE_FREQUENCY == 0:
            actions_sample = self.actor_local(states)
            actor_loss = -self.critic_1_local(states, actions_sample).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.actor_local, self.actor_target, TAU)
            self.soft_update(self.critic_1_local, self.critic_1_target, TAU)
            self.soft_update(self.critic_2_local, self.critic_2_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self):
        """Save the model weights"""
        torch.save(self.actor_local.state_dict(), 'actor_checkpoint.pth')
        torch.save(self.critic_1_local.state_dict(), 'critic_1_checkpoint.pth')
        torch.save(self.critic_2_local.state_dict(), 'critic_2_checkpoint.pth')
    
    def _print(self):
        """Print the hyperparameters used to train the agent"""
        msg = ("HYPERPARAMETERS:\n---------------\n\n"
              "BUFFER_SIZE = {0}\n"
              "BATCH_SIZE = {1}\n"
              "GAMMA = {2}\n"
              "TAU = {3}\n"
              "LR_ACTOR = {4}\n"
              "LR_CRITIC_1 = {5}\n"
              "LR_CRITIC_2 = {5}\n"
              "UPDATE_EVERY = {6}\n".format(BUFFER_SIZE,
                                            BATCH_SIZE,
                                            GAMMA,
                                            TAU,
                                            LR_ACTOR,
                                            LR_CRITIC_1,
                                            LR_CRITIC_2,
                                            UPDATE_EVERY))
        print(msg)
        

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device for value placement during training, either gpu or cpu (e.g. `cuda:0`)
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class GaussianNoise(object):
    """Gaussian Noise with optional clipping"""
    def __init__(self, action_size, sigma=0.1, clip_value=0.1):
        """Initialize parameters and noise process."""
        self.mu = torch.zeros(action_size)
        self.sigma = sigma
        self.clip_value = clip_value

    def __call__(self, clip):
        """Return (clipped) gaussian noise"""
        x = torch.normal(self.mu, self.sigma)
        if clip:
            x = x.clamp(-self.clip_value, self.clip_value)
        return x
