#%%
from MCTS import MCTS
from MCTS import Node
import gym #OpenAI library
import gym_chess #ChessAlphaZero implementation
import numpy as np

#Dummy NN just for testing, and code repairing
import torch
import torch.nn as nn
from AlphaZero import DummyNN




#%% Testing NN
'''
dummy_model = DummyNN(64)

env = gym.make("ChessAlphaZero-v0")
a = env.reset()

tensor_state = torch.tensor(a.astype(np.float32))
tensor_state= torch.permute(tensor_state,(2,0,1))
tensor_state = tensor_state.unsqueeze(0)
print(tensor_state.shape)
dummy_model.eval()
policy, value =dummy_model(tensor_state)
value = value.item()
policy = policy.squeeze(0).detach().numpy()
'''

#%% #Testing MCTS for chess
dummy_model = DummyNN(64)
env = gym.make("ChessAlphaZero-v0")
mcts = MCTS(env,4,"rollouts",10,neural_network=dummy_model,dirichlet_alpha=1)
initial_state = env.reset()
node1 = Node(initial_state,None)
action_taken=None
terminated=False
while not terminated:
    node1,p1 = mcts.search_tree(node1,True)
    action_taken = node1.action
    new_state,reward,terminated,info = env.step(action_taken)
    print(env.render(mode='unicode'))
    print("----------")
    
print(reward)
print("Nice")