# %%
from IPython import get_ipython

# %% [markdown]
# # DefensePractice-v0
# ---
# In this notebook, you will implement a DDPG agent with OpenAI Gym's MountainCarContinuous-v0 environment.
# 
# ### 1. Import the Necessary Packages

# %%
import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from Agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

from defense_environment import DefensePractice

# %% [markdown]
# ### 2. Instantiate the Environment and Agent
# 
# Initialize the environment in the code cell below.

# %%
random_seed = 7

# points to win
win_points = 1000
print('Red or Blue need a score of {} to win'.format(win_points))

#env = gym.make('MountainCarContinuous-v0')
env = DefensePractice()
env.seed(random_seed)

# size of each action
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)

# examine the state space 
state_size = env.observation_space.shape[0]
print('Size of state:', state_size)

action_low = env.action_space.low
print('Action low:', action_low)

action_high = env.action_space.high
print('Action high: ', action_high)

# %% [markdown]
# ### 3. Train the Agent with DDPG
# 
# Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!

# %%
from itertools import count
import time

agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)


# %%
def save_model():
    print("Model Save...")
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')


# %%
def ddpg(n_episodes=100000, max_t=1500, print_every=1, save_every=20):
    red_scores_deque = deque(maxlen=print_every)
    red_scores = []
    
    blue_scores_deque = deque(maxlen=print_every)
    blue_scores = []
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        red_state = state[0]
        blue_state = state[2]
        agent.reset()
        red_score = 0
        blue_score = 0
        timestep = time.time()
        for t in range(max_t):
            blue_action = agent.act(blue_state)
            red_action = agent.act(red_state)
            env.render()
            next_red_state, red_reward, next_blue_state, blue_reward, done, _ = env.step(np.concatenate([red_action, blue_action]))
            agent.step(blue_state, blue_action, blue_reward, next_blue_state, done, t)
            agent.step(red_state, red_action, red_reward, next_red_state, done, t)
            red_score += red_reward
            blue_score += blue_reward
            red_state = next_red_state
            blue_state = next_blue_state
            if done:
                break 
                
        red_scores_deque.append(red_score)
        red_scores.append(red_score)
        red_score_average = np.mean(red_scores_deque)

        blue_scores_deque.append(blue_score)
        blue_scores.append(blue_score)
        blue_score_average = np.mean(blue_scores_deque)
        
        if i_episode % save_every == 0:
            save_model()
        
        if i_episode % print_every == 0:
            print('\r[RED]  Episode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'.format(i_episode, red_score_average, np.max(red_scores), np.min(red_scores), time.time() - timestep), end="\n")
                    
        if np.mean(red_scores_deque) >= win_points:            
            save_model()
            print('Red won in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, red_score_average))            
            break       

        if i_episode % print_every == 0:
            print('\r[BLUE] Episode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'.format(i_episode, blue_score_average, np.max(blue_scores), np.min(blue_scores), time.time() - timestep), end="\n")
                    
        if np.mean(blue_scores_deque) >= win_points:            
            save_model()
            print('Blue won in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, blue_score_average))            
            break            
    
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%%
env.close()

# %% [markdown]
# ### 4. Watch a Smart Agent!
# 
# In the next code cell, you will load the trained weights from file to watch a smart agent!

# %%
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

for _ in range(5):
    state = env.reset()
    red_state = state[0]
    blue_state = state[2]
    for t in range(1200):
        red_action = agent.act(red_state)
        blue_action = agent.act(blue_state)
        env.render()
        red_state, red_reward, blue_state, blue_reward, done, _ = env.step(np.concatenate([red_action, blue_action]))
        if done:
            break 

env.close()


# %%


