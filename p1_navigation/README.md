# Project 1: Navigation

### Introduction

This Reinforcement Learning (RL) project contains an example of a DQN to solve the unity banana collection environment. 

### Models and Architectures

This DQN uses methods to improve performance such as:

- Soft-update fixed targets
- Experience Replay

There is also a separate dueling DQN implementation in `model.py` (`DuelingQNetwork`).

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

The `Banana.exe` file and `Banana_Data` folder are necessary for the unity environment to run, but require no installation.

### Instructions

Start a jupyter server from the command line or pycharm, and open `Navigation.ipynb`. Run each code cell to see the agent train. Hyperparameters for training the agent are set in the `dqn_agent.py` file and can be adjusted as needed. Explanations are provided where applicable. Some RL knowledge is assumed.

### Acknowledgements

The data and banana environment for this project was provided by Udacity.