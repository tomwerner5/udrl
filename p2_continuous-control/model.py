import torch
import torch.nn as nn


# class Actor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed):
#         """Initialize parameters and build model.
#
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#         """
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.state_size = state_size
#         self.action_size = action_size
#
#         self.h1 = nn.Linear(state_size, 64)
#         self.h2 = nn.Linear(64, 64)
#         self.h3 = nn.Linear(64, action_size)
#
#     def forward(self, state):
#         """Network that maps state -> action values.
#
#         Params
#         ======
#             state (array): state values to pass to the network
#         """
#         x = torch.relu(self.h1(state))
#         x = torch.relu(self.h2(x))
#         x = torch.tanh(self.h3(x))
#         return x
#
#
# class Critic(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#         """
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.state_size = state_size
#         self.action_size = action_size
#
#         # input/hidden layer
#         self.h1_a = nn.Linear(action_size, 64)
#         self.h1_s = nn.Linear(state_size, 32)
#
#         # concat layer
#         self.h2_c = nn.Linear(64+32, 64)
#
#         # output layer
#         self.h2_out = nn.Linear(64, 1)
#
#     def forward(self, state, action):
#         """Network that maps state -> action values.
#
#         Params
#         ======
#             state (array): states to pass to the network
#             action (array): actions to pass to the network
#         """
#         xs = torch.relu(self.h1_s(state))
#         xa = torch.relu(self.h1_a(action))
#         x = torch.relu(self.h2_c(torch.cat((xs, xa), dim=1)))
#         value = self.h2_out(x)
#
#         return value



############### This got there #################
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 128  # minibatch size
# GAMMA = 0.99  # discount factor
# TAU = 1e-3  # for soft update of target parameters
# LR_ACTOR = 1e-4  # learning rate
# LR_CRITIC = 1e-4  # learning rate
# UPDATE_EVERY = 1  # how often to update the network
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.h1 = nn.Linear(state_size, 64)
        self.h2 = nn.Linear(64, 32)
        self.h3 = nn.Linear(32, action_size)

    def forward(self, state):
        """Network that maps state -> action values.

        Params
        ======
            state (array): state values to pass to the network
        """
        x = torch.relu(self.h1(state))
        x = torch.relu(self.h2(x))
        x = torch.tanh(self.h3(x))
        return x

    def _print(self):
        print('_'.join([str(p.shape) for p in self.parameters()]))



class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # input/hidden layer
        self.h1_a = nn.Linear(action_size, 32)
        self.h1_s_1 = nn.Linear(state_size, 64)
        self.h1_s_2 = nn.Linear(64, 128)

        # concat layer
        self.h2_c = nn.Linear(128+32, 128)

        # output layer
        self.h2_out_1 = nn.Linear(128, 64)
        self.h2_out_2 = nn.Linear(64, 1)

    def forward(self, state, action):
        """Network that maps state -> action values.

        Params
        ======
            state (array): states to pass to the network
            action (array): actions to pass to the network
        """
        xs = torch.relu(self.h1_s_1(state))
        xs = torch.relu(self.h1_s_2(xs))
        xa = torch.relu(self.h1_a(action))
        x = torch.relu(self.h2_c(torch.cat((xs, xa), dim=1)))
        x = torch.relu(self.h2_out_1(x))
        value = self.h2_out_2(x)

        return value

    def _print(self):
        print('_'.join([str(p.shape) for p in self.parameters()]))
