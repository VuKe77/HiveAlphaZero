#%%
from MCTS import MCTS
from MCTS import Node
import FlashO1
import numpy as np

#Dummy NN just for testing, and code repairing
import torch
import torch.nn as nn
from Model_architecture import HiveAlphaZeroModel 






#%% Testing NN
''''
dummy_model = HiveAlphaZeroModel()


env = FlashO1.FlashO1()
env.startAs(0)
state = np.array(env.getBoardState())

tensor_state = torch.tensor(state.astype(np.float32))
tensor_state= torch.permute(tensor_state,(2,0,1))
tensor_state = tensor_state.unsqueeze(0)
print(tensor_state.shape)
dummy_model.eval()
policy, value =dummy_model(tensor_state)
value = value.item()
policy = policy.squeeze(0).detach().numpy()
'''
#%%


#%% #Testing MCTS for chess

dummy_model = HiveAlphaZeroModel()
env = FlashO1.FlashO1()

mcts = MCTS(env,4,"rollouts",2000,neural_network=dummy_model,dirichlet_alpha=1)
env.startAs(0)
initial_state = env.getBoardState()
node1 = Node(initial_state,None)
action_taken=None

terminated=False
while not terminated:
    node1,p1 = mcts.search_tree(node1,True)
    action_taken = node1.action
    if action_taken ==-1:
        env.stepPass()
    else:
        env.step(action_taken) 
    terminated = env.getGameStatus()

print(terminated)
print("Nice")
