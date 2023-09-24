#%%
import os.path
from utils import Utils

#Create path to the model
dir_path = os.path.join(os.getcwd(),"trained_on_server")
model_path  = os.path.join(dir_path,"model_23_09_13_26_19") #Just change this part for wanted model!
Utils._plot_loss_curve(model_saving_path=model_path)

#%%
#Try out model by playing game against random agent, and plotting policy given out by model in every model turn
import torch
import gym
import gym_chess
import numpy as np
import random
import matplotlib.pyplot as plt
from MCTS import MCTS
from MCTS import Node
from Model_architecture import HiveAlphaZeroModel
from FlashO1 import FlashO1

model = HiveAlphaZeroModel()
model.load_state_dict(torch.load(os.path.join(model_path,"best_model")))
model.eval()
game = FlashO1()
game.startAs(6) #TODO: Look what piece will you place here! (6 should be Grasshopper)
current_state = np.array(game.getBoardState())
terminated= False
player = '1'

#%%

plt.figure()
move_cnt=0
while not terminated:
    if move_cnt==100:
        terminated= 3
    if player=='1':
        #use NN to get next action
        tensor_state = torch.tensor(current_state.astype(np.float32))
        tensor_state = torch.permute(tensor_state,(2,0,1)).unsqueeze(0)
        policy, value =model(tensor_state)
        policy = torch.nn.functional.softmax(policy,dim=1)
        value = value.item()
        policy = policy.squeeze(0).detach().numpy()
         #Visualize
    
        plt.clf()
        plt.plot(policy)
        plt.show()

        #Mask out invalid actions and normilize policy vector probabilities
        legal_moves = game.getLegalMoves()
        valid_actions = [0]*3388
        for action in legal_moves :
            valid_actions[action]=1
        policy*=valid_actions
        policy=policy/(np.sum(policy)+1e-10)

        #find highest prob action
        action_taken = np.argmax(policy)
        #Sometimes it can happen that masked policy had 0 values, so wrong action would be taken
        if action_taken not in legal_moves:
            #Radnom move
            action_taken = random.choice(legal_moves)
            print("Took random action")
        player='2'
       

    elif player=='2':
        #Random move
        legal_moves = game.getLegalMoves()
        action_taken = random.choice(legal_moves)
        player='1'

    if len(legal_moves)==0:
        game.stepPass()
    else:
        game.step(action_taken)
    
    current_state = np.array(game.getBoardState())
    terminated = game.getGameStatus()
    move_cnt+=1
    
print(terminated)

#%% Perform game against random with MCTS:
'''
mcts = MCTS(game,4,"rollouts",budget = 1000,neural_network=model,dirichlet_alpha=1)
random_mcts =MCTS(game,4,"rollouts",budget = 1,neural_network=None,dirichlet_alpha=1)
terminated = False
initial_state = game.reset()
root = Node(initial_state,None)
random_root = Node(initial_state,None)
player = '1'
while not terminated:

    if player=='1':
        root = mcts.truncate_tree(root,random_root)
        root, action_prob = mcts.search_tree(root,print_tree=True)
        action_taken = root.action
        player='2'

    elif player=='2':
        random_root = random_mcts.truncate_tree(random_root,root)
        random_root, action_prob = random_mcts.search_tree(random_root,print_tree=True)
        action_taken = root.action
        player='1'

    new_state,game_outcome,terminated,info = game.step(root.action)
    print(game.render(mode='unicode'))
    print("----------")


# %%
'''