import torch
import torch.nn as nn


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

        self.h1 = nn.Linear(state_size, 84)
        self.h2 = nn.Linear(84, 32)
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

        # input/hidden layer for action space
        self.h1_a = nn.Linear(action_size, 32)

        # input/hidden layers for state space
        self.h1_s_1 = nn.Linear(state_size, 84)
        self.h1_s_2 = nn.Linear(84, 256)

        # concat layer
        self.h2_c = nn.Linear(256+32, 400)

        # output layer
        self.h2_out_1 = nn.Linear(400, 200)
        self.h2_out_2 = nn.Linear(200, 1)

    def forward(self, state, action):
        """Network that maps state -> action values.

        Params
        ======
            state (array): states to pass to the network
            action (array): actions to pass to the network
        """
        act = torch.nn.LeakyReLU(0.1)
        xs = act(self.h1_s_1(state))
        xs = act(self.h1_s_2(xs))
        xa = act(self.h1_a(action))
        x = act(self.h2_c(torch.cat((xs, xa), dim=1)))
        x = act(self.h2_out_1(x))
        value = self.h2_out_2(x)

        return value
