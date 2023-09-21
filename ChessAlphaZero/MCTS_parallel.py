import ray
from MCTS import Node
from MCTS import MCTS
from AlphaZero import DummyNN
import gym
import gym_chess
from copy import deepcopy
import time
import numpy as np
import sys
import psutil
import gc


#print(psutil.virtual_memory())


ray.init()
print(ray.available_resources())
print(psutil.virtual_memory())

dummy_model = DummyNN(1,64)
env = gym.make("ChessAlphaZero-v0")
#mcts = MCTS(env,4,"rollouts",1000,neural_network=dummy_model,dirichlet_alpha=1)
@ray.remote
def foo():
    time.sleep(10)
    return 1
@ray.remote
def self_play(game_id,game_env,dummy_model_ref):
    print(f"Game:{game_id}")
    game_copy = deepcopy(game_env)
    initial_state = game_copy.reset()
    mcts = MCTS(game_copy,4,"rollouts",100,neural_network=dummy_model_ref,dirichlet_alpha=1)
    
    root = Node(initial_state,None)
    turn_cnt =0
    one_game_data = []
    terminated = False
    while not terminated:
        #Playout the game and save game trajectories
        
        if turn_cnt==40:
            terminated = True
            break #early termination 

        old_state = root.state
        player = root.state[:,:,112][0][0] #TODO: Specific for Chess, rework for another game
        root, action_prob = mcts.search_tree(root)
        new_state,game_outcome,terminated,info = game_copy.step(root.action)

        one_game_data.append([old_state,action_prob,player]) #We are currently saving player in place of reward, when game is over we replace it with reward
        turn_cnt+=1
        #print(f"GAME INDEX:{index}:")
        #print(game_copy.render(mode='unicode'))


    #When game is over add reward to one_game_data instead of player => P1 win(+1), P2 win(-1), draw(+0)
    #TODO: specific for Chess
    
    for data in one_game_data:
        if game_outcome==1: #white player wins
            if data[2]==1: #white was on move
                data[2]=1
            else: #black was on move
                data[2]=-1
        elif game_outcome ==-1: #black player wins
            if data[2]==1:#white was on move
                data[2]=-1 
            else:#black was on move
                data[2]=-1
        elif game_outcome==0: #draw
            data[2]=0
        else:
            raise ValueError("Uncategorized winner!")
            
    return one_game_data




# param_size = 0
# for param in dummy_model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in dummy_model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2

#print('model size: {:.3f}MB'.format(size_all_mb))
BATCH_SIZE = 10
WORKERS = 4

for i in range(5):
    not_ready =[] 
    ready=[]
    data = []
    dummy_model_ref = ray.put(dummy_model)
    games_played = 0
    while(games_played<BATCH_SIZE):
        if len(not_ready)>=WORKERS:
            ready, not_ready = ray.wait(not_ready,num_returns=1)
            data.append(deepcopy(ray.get(ready[0])))


        not_ready.append(self_play.remote(games_played+1,env,dummy_model_ref))
        games_played+=1

    for i in range(WORKERS):
        data.append(deepcopy(ray.get(not_ready[i])))
    #test = [foo.remote() for i in range(7)]
    del ready
    del not_ready
    del dummy_model_ref
    

for i in range(10):
    print(i)
    time.sleep(10)
print("Kraj")




# initial_state = env.reset()
# node1 = Node(initial_state,None)
# action_taken=None
# terminated=False
# while not terminated:
#     node1,p1 = mcts.search_tree(node1,True)
#     action_taken = node1.action
#     new_state,reward,terminated,info = env.step(action_taken)
#     print(env.render(mode='unicode'))
#     print("----------")
    
# print(reward)
# print("Nice")