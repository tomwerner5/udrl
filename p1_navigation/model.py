import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.h1 = nn.Linear(state_size, 64)
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Network that maps state -> action values.

        Params
        ======
            state (array): state values to pass to the network
        """
        x = torch.relu(self.h1(state))
        x = torch.relu(self.h2(x))
        x = self.h3(x)
        return x


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # input/hidden layer
        self.h1 = nn.Linear(state_size, 64)

        # value stream
        self.h2_value_1 = nn.Linear(64, 64)
        self.h2_value_out = nn.Linear(64, 1)

        # advantage stream
        self.h2_advantage_1 = nn.Linear(64, 64)
        self.h2_advantage_out = nn.Linear(64, action_size)

    def forward(self, state):
        """Network that maps state -> action values.

        Params
        ======
            state (array): state values to pass to the network
        """
        x = torch.relu(self.h1(state))
        value = torch.relu(self.h2_value_1(x))
        value = self.h2_value_out(value)

        advantage = torch.relu(self.h2_advantage_1(x))
        advantage = self.h2_advantage_out(advantage)

        action_value = value + advantage - advantage.mean(dim=1, keepdim=True)
        return action_value
