#%%
from TicTacToe import TicTacToe
from MCTS import MCTS
from MCTS import Node
from AlphaZero import ResNet

"""
THIS FILE IS USED FOR TESTING!
"""


def get_human_input():
    """Print a list of legal next states for the human player, and return
    the player's selection.
    """
    legal_next_states = game_env.legal_next_states
    for idx, state in enumerate(legal_next_states):
        print(state[1], '\t', idx, '\n')
    move_idx = int(input('Enter move index: '))
    game_env.step(legal_next_states[move_idx])
    return legal_next_states[move_idx]


#%% Test out MCTS on TicTacToe game
'''
game_env = TicTacToe()
initial_state = game_env.state
root1 = Node(initial_state,None)
root2 = Node(initial_state,None)
model = ResNet(game_env,4,64)
MCTS = MCTS(game_env,4,"rollouts",budget = 2000,neural_network=model,dirichlet_alpha=1)

game_env.print_board()
#MCTS = MCTS(game_env,1.2,"rollouts",1000)

#Test without neural network
node1 = root1
node2 = root2
player = '1'

#Game where we have one MC agent playing vs another MC agent
'''
'''
while not game_env.outcome:
    if player=='1':
        node1 = MCTS.truncate_tree(node1)
        #old = node1.state##debug
        node1,a1 = MCTS.search_tree(node1,True)
        
        player = '2'
        choice = node1.state    
        
    else:
        # human_move = get_human_input()
        # player = '1'
        
        node2 = MCTS.truncate_tree(node2)
        #old = node2.state##debug
        node2,a2 = MCTS.search_tree(node2,True)
        
        player = '1'
        choice = node2.state
        

    game_env.step(choice)
    game_env.print_board()
    #a=MCTS._action_mapping(old,choice)#debug
'''
#%% One MCTS against itself

'''
initial_state = game_env.state
root3 = Node(initial_state,None)
while not game_env.outcome:
    root3 = MCTS.truncate_tree(root3)
    root3,a3 = MCTS.search_tree(root3, True)
    choice = root3.state
    game_env.step(choice)
    game_env.print_board()

game_env.reset()
game_env.print_board()
'''

#%%
#Testing out alphaZero for TicTacToe
import matplotlib.pyplot as plt

import torch
import numpy as np
from TicTacToe import TicTacToe

game_env = TicTacToe()
model = ResNet(game_env,4,64)

played_moves = []
plt.figure()
game_env.reset()
game_env.print_board()
model.load_state_dict(torch.load('test_model')) #HERE ENTER MODEL YOU WANT TO TEST
model.eval()

MCTS = MCTS(game_env,4,"rollouts",budget = 2000,neural_network=None,dirichlet_alpha=1)
#BUDGET OF 2000 PERFORMS PERFECT
node1 = Node(None,None)

#Model doesn't use MCTS at all for this test, just raw outputs of NN
player ='2'#Set starting player
while not game_env.outcome:
    #for i in range(5):
    plt.clf()
    game_env.print_board()
    if player=='1':
        tensor_state = torch.tensor(game_env.state.astype(np.float32)).unsqueeze(0)
        policy, value =model(tensor_state)
        value = value.item()
        policy = policy.squeeze(0).detach().numpy()
        
        plt.bar(range(9),policy)
        plt.title(value)
        plt.show()

        
        target_index = np.argmax(policy)
        while target_index in played_moves:
            #IF move is already played find next one with biggest probability
            policy[target_index]=0
            target_index=np.argmax(policy)
        if game_env.current_player(game_env.state) == 'player1':
            next_state = game_env.state
            next_state[0][target_index//3][target_index%3]=1
            next_state[2] = np.ones((3,3))
        elif game_env.current_player(game_env.state) == 'player2':
            next_state = game_env.state
            next_state[1][target_index//3][target_index%3]=1
            next_state[2] = np.zeros((3,3))
        played_moves.append(target_index)
        #print(next_state)
        game_env.step(next_state)
        
        player='2'
    elif player=='2':
         #human_move = get_human_input()
         #player = '1'
         node1 = MCTS.truncate_tree(node1)

         node1,a1 = MCTS.search_tree(node1,True)
        
         player = '1'
         choice = node1.state  
         game_env.step(choice)
    game_env.print_board()

# %%
