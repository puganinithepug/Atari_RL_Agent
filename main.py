#!/usr/bin/env python
# coding: utf-8

# In[9]:


import gymnasium as gym
import ale_py
import importlib
import model
import numpy as np

importlib.reload(model)
from model import PolicyGradient

# Initialise the environment
#https://ale.farama.org/environments/ wants Atari from here
gym.register_envs(ale_py)

env = gym.make("ALE/AirRaid-v5", render_mode="human")
# , render_mode="human" add this later

#num_features = env.observation_space.shape[0]
#num_features = env.observation_space.shape[0]
num_features = np.prod(env.observation_space.shape) 
num_actions = env.action_space.n

m = PolicyGradient(num_features, num_actions)
print(m)

states, actions, rewards = [], [], []

# action space: Discrete(6)
# state space: Box(0, 255, (250, 160, 3), uint8)

# Reset the environment to generate the first observation
s, info = env.reset(seed=42)
done = False

# an episode = from moment of start to death

while not done:
    # this is where you would insert your policy
    #a = env.action_space.sample()
    #a = m.get_action(s)
    a = m.get_action(s.flatten())

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    nxt_s, r, terminated, truncated, info = env.step(a)
    env.render()

    #collect data set for training 
    #states.append(s)
    states.append(s.flatten())
    actions.append(a)
    rewards.append(r)

    s = nxt_s

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        done = True
print(rewards)        

# train model on data set
m.train_step(np.array(states), np.array(actions), np.array(rewards, dtype=np.float32))

env.close()

