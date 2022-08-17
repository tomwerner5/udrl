# Project 2: Continuous Control

### Introduction

This Reinforcement Learning (RL) project contains an example of a Twin Delayed DDPG (TD3) to solve the unity Reacher environment. This project uses Pytorch for the models and Unity for the environment rendering.

### Contents

- *Reacher_Data*: folder of necessary files for the custom unity environment
- *Reacher.exe*: Windows 64-bit executable of the unity environment
- *td3_agent.py*: Implementation of the agent used to make actions and learn
- *env.txt*: The requirements.txt or environment specs used to run/train
- *model.py*: The pytorch neural network implementations for the actors and critics
- *Continuous_Control.ipynb*: (Start Here after README) jupyter notebook used to implement and visualize algorithm
- *Report.md*: Summary of project outcomes

### Models and Architectures

This project uses a Twin Delayed Deep Deterministic Policy Gradient (TD3) to solve the environment, which use deep neural networks as function approximators for the actions taken by the agent (actor) and the value function (critic). For more information about TD3, see the original paper [here](https://arxiv.org/pdf/1802.09477.pdf). 

This TD3 is a vanilla implementation as described by the authors, utilizing basic concepts such as:

- Normal action noise with noise clipping for action selection and network updates
- Two critic networks (i.e. 'twins') and one actor network
- Delayed policy (actor) update
- Soft-update fixed targets (for all actor/critic target networks)
- Experience Replay Buffer

A detailed description of the models and architecture can be found in `Report.md`

![Trained Agent](trained_example.gif)

### Explanation of the RL Environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic and considered solved when the agent gets an average score of +30 over 100 consecutive episodes.

### Installation and Setup

The analysis consists of a jupyter notebook titled `Continuous_Control.ipynb`. To run the code, the following is required to be installed:

- Python==3.6.3
- unityagents==0.4.0
- torch==1.10.2

Alternatively, you can install the exact environment using the `env.txt` file by treating it as a `requirements.txt` file in pip.

The `Reacher.exe` file and `Reacher_Data` folder are necessary for the unity environment to run, but require no installation. However, the files in this repository assume that you are running on 64-bit Windows. If you are **NOT** running on 64-bit Windows, please download the appropriate banana environment files by following the instructions in [this](https://github.com/udacity/Value-based-methods/tree/main/p2_continuous-control) repository.

Place the downloaded files in the `p2_continuous-control` folder of this repository.

### Instructions

Start a jupyter server from the command line or pycharm, and open `Continuous_Control.ipynb`. Run each code cell to see the agent train. Hyperparameters for training the agent are set in the `td3_agent.py` file and can be adjusted as needed. Explanations are provided where applicable. Some RL knowledge is assumed.

### Acknowledgements

The data and reacher environment for this project was provided by Udacity.
