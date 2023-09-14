#%% TESTING OUT GYM ENVIRONMENT
import gym
import gym_chess
import random
import chess
from copy import deepcopy
#%%

env = gym.make("ChessAlphaZero-v0")
env.reset()
env2 = deepcopy(env)

#env.reset()
#%%
#env.action_space
print(env.render(mode='unicode'))

while True:
    terminated=False
    current_space= env.reset()
    while not terminated:
        action = random.choice(env.legal_actions)
        current_state,reward,terminated,info  = env.step(action)
        print('--------')
        print(env.render(mode='unicode'))
       # if b[1]!=0:
          
        #print(env.render(mode='unicode'))
        #
        