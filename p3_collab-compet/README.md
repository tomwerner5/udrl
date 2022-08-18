# Project 3: Collaboration and Competition

### Introduction

This Reinforcement Learning (RL) project contains an example of a multi-agent Twin Delayed DDPG (MATD3) to solve the unity Tennis environment. This project uses Pytorch for the models and Unity for the environment rendering.

### Contents

- *Tennis_Data*: folder of necessary files for the custom unity environment
- *Tennis.exe*: Windows 64-bit executable of the unity environment
- *td3_agent.py*: Implementation of the agent used to make actions and learn
- *env.txt*: The requirements.txt or environment specs used to run/train
- *model.py*: The pytorch neural network implementations for the actors and critics
- *Tennis.ipynb*: (Start Here after README) jupyter notebook used to implement and visualize algorithm
- *Report.md*: Summary of project outcomes

### Models and Architectures

This project uses a multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) to solve the environment, which use deep neural networks as function approximators for the actions taken by the agent (actor) and the value function (critic). For more information about MATD3, see the original paper [here](https://arxiv.org/pdf/1910.01465.pdf). 

This MATD3 is a vanilla implementation as described by the authors, utilizing basic concepts such as:

- Normal action noise with noise clipping for action selection and network updates
- Two critic networks (i.e. 'twins') and one actor network
- Delayed policy (actor) update
- Soft-update fixed targets (for all actor/critic target networks)
- Experience Replay Buffer

A detailed description of the models and architecture can be found in `Report.md`

### Explanation of the RL Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. 
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Installation and Setup

The analysis consists of a jupyter notebook titled `Tennis.ipynb`. To run the code, the following is required to be installed:

- Python==3.6.3
- unityagents==0.4.0
- torch==1.10.2

Alternatively, you can install the exact environment using the `env.txt` file by treating it as a `requirements.txt` file in pip.

The `Tennis.exe` file and `Tennis_Data` folder are necessary for the unity environment to run, but require no installation. However, the files in this repository assume that you are running on 64-bit Windows. If you are **NOT** running on 64-bit Windows, please download the appropriate banana environment files by following the instructions in [this](https://github.com/udacity/Value-based-methods/tree/main/p3_collab-compet) repository.

Place the downloaded files in the `p2_collab-compet` folder of this repository.

### Instructions

Start a jupyter server from the command line or pycharm, and open `Tennis.ipynb`. Run each code cell to see the agent train. Hyperparameters for training the agent are set in the `td3_agent.py` file and can be adjusted as needed. Explanations are provided where applicable. Some RL knowledge is assumed.

### Acknowledgements

The data, starter code, and Tennis environment for this project was provided by Udacity.