# Project 1: Navigation

### Introduction

This Reinforcement Learning (RL) project contains an example of a DQN to solve the unity banana collection environment. This project uses Pytorch for the models and Unity for the environment rendering.

### Contents

- *Banana_Data*: folder of necessary files for the custom unity environment
- *Banana.exe*: Windows 64-bit executable of the unity environment
- *dqn_agent.py*: Implementation of the agent used to make actions and learn
- *env.txt*: The requirements.txt or environment specs used to run/train
- *model.py*: The pytorch neural network implementations (DQN and Dueling DQN)
- *Navigation.ipynb*: (Start Here after README) jupyter notebook used to implement and visualize DQN algorithm
- *Report.md*: Summary of project outcomes

### Models and Architectures

This project uses a Deep Q-Network (DQN) to solve the environment, which use deep neural networks to as a function approximator to the action-value function in Q-learning. For more information about DQNs, see the original paper [here](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). 

This DQN uses methods to improve performance such as:

- Soft-update fixed targets
- Experience Replay

There is also a separate dueling DQN implementation in `model.py` (`DuelingQNetwork`).

A detailed description of the models and architecture can be found in `report.ipynb`

![Trained Agent](trained_example.gif)

### Explanation of the RL Environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic and considered solved when the agent gets an average score of +13 over 100 consecutive episodes. An example of a solved, trained agent is provided in the `checkpoint.pth` file, which can be loaded into an existing torch neural network object using `torch_network.load_state_dict(torch.load('checkpoint.pth'))`.

### Installation and Setup

The analysis consists of a jupyter notebook titled `Navigation.ipynb`. To run the code, the following is required to be installed:

- Python==3.6.3
- unityagents==0.4.0
- torch==1.10.2

Alternatively, you can install the exact environment using the `env.txt` file by treating it as a `requirements.txt` file in pip.

The `Banana.exe` file and `Banana_Data` folder are necessary for the unity environment to run, but require no installation. However, the files in this repository assume that you are running on 64-bit Windows. If you are **NOT** running on 64-bit Windows, please download the appropriate banana environment files by following the instructions in [this](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation) repository.

Place the downloaded files in the `p1_navigation` folder of this repository.

### Instructions

Start a jupyter server from the command line or pycharm, and open `Navigation.ipynb`. Run each code cell to see the agent train. Hyperparameters for training the agent are set in the `dqn_agent.py` file and can be adjusted as needed. Explanations are provided where applicable. Some RL knowledge is assumed.

### Acknowledgements

The data and banana environment for this project was provided by Udacity.