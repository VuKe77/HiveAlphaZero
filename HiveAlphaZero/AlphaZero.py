#%%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MCTS import MCTS
from MCTS import Node
import time
import random
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp
from ray.util.multiprocessing import Pool
import ray
import FlashO1


@ray.remote
def self_play(game_id,nn_ref,args_ref):

    print(f"Game:{game_id}")
    game_copy = FlashO1.FlashO1()
    game_copy.startAs(6) #TODO: Look what piece will you place here! (6 should be Grasshopper)
    initial_state = game_copy.getBoardState()
    mcts = MCTS(game_copy,UCT_c=args_ref["MCTS_UCT_c"],constraint=args_ref["MCTS_constraint"],
                budget=args_ref["MCTS_budget"],neural_network=nn_ref,dirichlet_alpha=args_ref["MCTS_dirichlet_alpha"])
    
    root = Node(initial_state,None)
    turn_cnt =0
    one_game_data = []
    terminated = 0 #0(running), 1(white wins), 2(black wins), 3(draw)
    while not terminated:
        #Playout the game and save game trajectories
        
        if turn_cnt==args_ref["max_moves_num_selfplay"]:
            terminated = 3 #draw
            break #early termination 

        old_state = root.state
        player = game_copy.isWhiteToPlay() #TODO: Specific for Hive, rework for another game
        root, action_prob = mcts.search_tree(root)

        if root.action==-1:
            game_copy.stepPass() 
        else:
            game_copy.step(root.action)

        terminated = game_copy.getGameStatus()
        
        #Save turn data
        one_game_data.append([old_state,action_prob,player]) #We are currently saving player in place of reward, when game is over we replace it with reward
        turn_cnt+=1
        #print(f"GAME INDEX:{index}:")
        #print(game_copy.render(mode='unicode'))


    #When game is over add reward to one_game_data instead of player => P1 win(+1), P2 win(-1), draw(+0)
    #TODO: specific for Hive
    
    for data in one_game_data:
        if terminated ==3: #draw
            data[2] = 0
        elif terminated==1: #white player wins
            if data[2]==True: #white was on move
                data[2]=1
            else: #black was on move
                data[2]=-1
        elif terminated ==2: #black player wins
            if data[2]==True:#white was on move
                data[2]=-1 
            else:#black was on move
                data[2]=1
        else:
            raise ValueError("Uncategorized winner!")
    return one_game_data

@ray.remote
def evaluate_model(game_id,new_nn_ref,old_nn_ref,args_ref):
    """
    Plays out game with new model and old model, to see which is better. Using
    args_ref in order to set up MCTS for evaluation.
    """
    #Set parameters for  evaluation with MCTS
    print(f"Game:{game_id}")
    game_copy =  FlashO1.FlashO1()
    mcts = MCTS(game_copy,UCT_c=args_ref["MCTS_UCT_c"],constraint=args_ref["MCTS_constraint"],
                budget=args_ref["MCTS_budget"],neural_network=new_nn_ref,dirichlet_alpha=args_ref["MCTS_dirichlet_alpha"])
    
    game_copy.startAs(6) #TODO: Look what piece will you place here! (2 should be grasshopper)
    initial_state = game_copy.getBoardState()
    node1 = Node(initial_state,None)
    node2 = Node(initial_state,None)

    #select starting player
    if game_id%2==0:
        player = '1' #old model plays first
    else:
        player = '2' #new model plays first

    #Playout games
    move_cnt=0
    terminated = 0
    while not terminated:
        if move_cnt==args_ref["max_moves_num_eval"]:
            terminated=3
            break

        if player=='1':
            mcts.set_new_model(old_nn_ref) #set old model for MCTs
            node1 = mcts.truncate_tree(node1,node2)
            node1,a1 = mcts.search_tree(node1)
            player = '2'
            action_taken = node1.action
            
        elif player =='2':
            mcts.set_new_model(new_nn_ref) #set new model for MCTs
            node2 = mcts.truncate_tree(node2,node1)
            node2,a2 = mcts.search_tree(node2)
            player = '1'
            action_taken = node2.action

        if action_taken==-1:
            game_copy.stepPass() 
        else:
            game_copy.step(action_taken)

        terminated = game_copy.getGameStatus()
        move_cnt+=1

    #Check what NN won the game:
    if terminated == 3: #draw
        return 0
    elif terminated ==1: #white wins
        if game_id%2 ==0: #old model is white 
            return -1
        else:
            return 1
    elif terminated == 2: #black wins
        if game_id%2 ==0: #old model white
            return 1
        else:
            return -1
    else:
        raise ValueError("Invalid outcome!") 

@ray.remote
def play_against_random(game_id,nn_ref,args_ref):

    print(f"Game:{game_id}")
    game_copy = FlashO1.FlashO1()
    game_copy.startAs(6) #TODO: Look what piece will you place here! (6 should be Grasshopper)
    initial_state = game_copy.getBoardState()
    mcts = MCTS(game_copy,UCT_c=args_ref["MCTS_UCT_c"],constraint=args_ref["MCTS_constraint"],
                budget=args_ref["MCTS_budget"],neural_network=nn_ref,dirichlet_alpha=args_ref["MCTS_dirichlet_alpha"])
    
    node1 = Node(initial_state,None)
    randomAgent_action = None
    new_state=None
    #select starting player
    if game_id%2==0:
        player='1' #first_player = "NN"
           
    else:
        player='2' #first_player="RANDOM"
           
    move_cnt =0
    terminated = 0 #0(running), 1(white wins), 2(black wins), 3(draw)
    while not terminated:
        if move_cnt==args_ref["max_moves_num"]:
            terminated=3
            break

        if player=='1':
            node1 = mcts.truncate_tree(node1,randomAgent_action,playing_against_random=True,new_state=new_state)
            node1,a1 = mcts.search_tree(node1)
            action_taken = node1.action
            player = '2'
            
        elif player =='2':
            legal_moves = game_copy.getLegalMoves()
            if not legal_moves:
                action_taken = -1 #Pass if no legal moves are available
            else:
                action_taken = random.choice(legal_moves)

            randomAgent_action = action_taken
            player = '1'

        #Do step in the game
        if action_taken==-1:
            game_copy.stepPass()
        else:
            game_copy.step(action_taken)
        terminated = game_copy.getGameStatus()
        new_state = game_copy.getBoardState()
        move_cnt+=1
    #determine reward
    if terminated==3: #draw
        return 0

    elif terminated==1:#white wins
        if game_id%2==0: #White player is NN
            return 1
        else:
            return -1
    elif terminated ==2:#black wins
        if game_id%2==1:#white player is random
            return 1
        else:
            return -1
    else:
        raise ValueError("Uncategorized winner")
    






class AlphaZero:
    def __init__(self,model, optimizer,cpu,gpu, args,model_saving_path):
        """
        Training pipeline class for AlphaZero alghorihm.
        """
        self.cpu_device = cpu
        self.gpu_device = gpu
        self.model = model
        self.old_model = None 
        self.optimizer = optimizer
        self.args = args
        self.model_saving_path = model_saving_path #Path for saving our model
        ''' ARGS:
        -MCTS_UCT_c: MCTS constant for UCB
        -MCTS_constraint: training constraint, can be "rollouts" or "time"
        -MCTS_budget: budget for every MCTS iteration, if constraint is rollouts: number of rollouts
        if constraint is time: time in seconds
        -MCTS_dirichlet_alpha: alpha used in dirichlet distribution
        -num_self_play_iterations:  number of self-play games performed in one cycle of learning 
        -max_moves_num_selfplay: constraint of maximum number of moves that can be performed in one self-play game. If achieved
        is achieved game is declared draw!
        -num_learning_iterations: number of learning iterations. One learning iteration consists of:
        self-play, training on self-play data, NN evaluation
        -num_epochs: number of swipes through data acquiered by self-pay in training phase
        -batch_size: size of one batch used for training data with mini-bathces
        -num_evaluation_games: number of evaluation games played in evaluation phase
        -num_games_against_random: number of games played against random agent in order to check reward

        ''' 
    def _parallel_self_play(self):
        """
        Performs number of self-play games defined in args["num_self_play_iterations"]. Every self-play game has early  
        termination if number of moves exceeds args["max_move_num"] . Returns vector containing every game trajectory, where  
        every move is in format [state of game, posterior action probs, reward]. Data acquisition is done in parallel using
        multiple python processes.
        """
        time1 = time.time()
        print("Self-playing phase starts...")
        #Utilising resources
        workers = round(mp.cpu_count()*0.5)
        if workers>self.args["num_self_play_iterations"]:
            workers = self.args["num_self_play_iterations"]
        print(f"Using {workers} CPU cores...")
        #Rounding batch size in order to fully utilise resources
        BATCH_SIZE = self.args["num_self_play_iterations"]//workers*workers #batch size is number of self played games

        #Run multiple processes and acquire data
        not_ready =[] 
        ready=[]
        data = []
        model_ref = ray.put(self.model)
        args_ref = ray.put(self.args)
        games_played = 0

        while(games_played<BATCH_SIZE):
            if len(not_ready)>=workers:
                ready, not_ready = ray.wait(not_ready,num_returns=1)
                data+=deepcopy(ray.get(ready[0]))

            not_ready.append(self_play.remote(games_played+1,model_ref,args_ref))
            games_played+=1

        #Get remaining  data
        for i in range(workers):
            data+=deepcopy(ray.get(not_ready[i]))
        #del ready
        #del not_ready
        #del dummy_model_ref
        #del game_ref
        print(f"Self-play finished. Played out {BATCH_SIZE} games.")
        print(f"Time needed for {BATCH_SIZE} games is: {time.time()-time1:.2f}s")
        return data

   
    def _train(self, memory):
        """
        Here is NN trained on memory data in batches of size args['batch_size'].
        Memory data is shuffled first. Returns average loss of training on memory
        as 2D vector(policy loss and value loss)

        """

        cumalative_loss=[[],[]] #Used for plotting loss curve, first array is policy lost, second is value loss

        #shuffle training data
        random.shuffle(memory)
        
        for batch_idx in range(0,len(memory),self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory),batch_idx+self.args['batch_size'])]
            states, policy_targets, value_targets = zip(*sample)
            states, policy_targets, value_targets = np.array(states), np.array(policy_targets), np.array(value_targets).reshape(-1,1) #TODO:need this?
            
            #Turn into tensors, prepare for NN
            states = torch.tensor(states,dtype=torch.float32)
            states =torch.permute(states,(0,3,1,2))
            policy_targets = torch.tensor(policy_targets,dtype=torch.float32)
            value_targets = torch.tensor(value_targets,dtype=torch.float32)

            #Move it to GPU
            states = states.to(self.gpu_device)
            policy_targets = policy_targets.to(self.gpu_device)
            value_targets = value_targets.to(self.gpu_device)
            out_policy, out_value = self.model(states)

            policy_loss = F.cross_entropy(out_policy,policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss+value_loss
            cumalative_loss[0].append(policy_loss.to(self.cpu_device).detach().numpy())  #used for plotting loss curve
            cumalative_loss[1].append(value_loss.to(self.cpu_device).detach().numpy())  #used for plotting loss curve

            #Backpropegate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.average(cumalative_loss,axis=1)

    def learn(self):
        """
        This functions performs AlphaZero algorithm. First old model is saved as best, than  
        number of self-play games are played to acquire data for learning. Training is performed in  
        epochs. At the end of learning phase, NN is evaluated by playing with old model.Number of times  
        this process is repeated is defined by 'num_learning_iterations'.
        """

        average_policy_loss_vector = [] #Policy loss is accumulated during training phase and than averaged and appended to this vector
        average_value_loss_vector = [] #Value loss is accumulated during training phase and than averaged and appended to this vector
        reward_history=[] #After every learning iterations games are played 10 against random agent, and final reward is appended to this vector
        self.model.eval()
        #print(next(model.parameters()).device)
        for iter in tqdm(range(self.args['num_learning_iterations'])):
            
            torch.save(self.model.state_dict(),os.path.join(self.model_saving_path,"best_model"))
            torch.save(self.optimizer.state_dict(),os.path.join(self.model_saving_path,"best_optimizer"))
          
            #Perform self-play to gather data for training 
            memory = [] #vector, whos elements are self-played games #TODO:Maybe put upper bound on memory limit
            memory+= self._parallel_self_play()
            
            
            #Train model on gathered data
            self.model.to(self.gpu_device)#Move model to GPU for training
            #print("Moved model to GPU")
            #print(next(model.parameters()).device)
            self.model.train()   
            print("Learning phase starts...")
            for epoch in range(self.args["num_epochs"]):
               average_loss = self._train(memory)
               average_policy_loss_vector.append(average_loss[0])
               average_value_loss_vector.append(average_loss[1])

            #Play with old model to evaluate NN
            self.model.to(cpu_device) #Move model to CPU for evaluation
            #print("Moved model to CPU")
            #print(next(model.parameters()).device)
            if self._evaluate_NN_parallel():
                #If new model is better we will playout games against random to see who's better
                #Play  games against random agent in order to evaluate network later
                reward = self._play_against_random_parallel()
                reward_history.append(reward)


            #save loss values and rewards during training   
            with open(os.path.join(self.model_saving_path,'loss_data.npz'),'wb') as f:
                np.savez(f,policy_loss=average_policy_loss_vector,value_loss = average_value_loss_vector, reward = reward_history)

            print("-----------------------")
        #Save best model at the end of the training
        torch.save(self.model.state_dict(),os.path.join(self.model_saving_path,"best_model"))
        torch.save(self.optimizer.state_dict(),os.path.join(self.model_saving_path,"best_optimizer"))
       

    def _evaluate_NN_parallel(self):
        """
        Evaluates model, playing args[num_evaluation_games] in parallel with old model.
        Returns True if new model is better, otherwise returns False and sets old model as best model and uses it 
        in next learning iteration..
        """
        new_model_better = True
        if self.args["num_evaluation_games"]==0:
            return new_model_better
        
        print("NN evaluation phase starts...")
        time1 = time.time()
        #Utilising resources
        workers = round(mp.cpu_count()*0.3)
        if workers>self.args["num_evaluation_games"]:
            workers = self.args["num_evaluation_games"]
        print(f"Using {workers} CPU cores...")
        BATCH_SIZE = self.args["num_evaluation_games"]

        #Load old model, prepare for evaluation
        self.old_model =  deepcopy(self.model)
        self.old_model.load_state_dict(torch.load(os.path.join(self.model_saving_path,"best_model")))
        self.old_model.eval()
        self.model.eval()

        #Arguments for MCTS used for evaluation
        mcts_args ={
        "MCTS_UCT_c" : 4,
        "MCTS_constraint": "rollouts",
        "MCTS_budget": 10,
        "MCTS_dirichlet_alpha": 1,
        "max_moves_num_eval": 60,
        }

        #Run multiple processes and acquire data
        not_ready =[] 
        ready=[]
        rewards = []
        new_model_ref = ray.put(self.model)
        old_model_ref = ray.put(self.old_model)
        args_ref = ray.put(mcts_args)
        games_played = 0

        while(games_played<BATCH_SIZE):
            if len(not_ready)>=workers:
                ready, not_ready = ray.wait(not_ready,num_returns=1)
                rewards.append(deepcopy(ray.get(ready[0])))

            not_ready.append(evaluate_model.remote(games_played+1,new_model_ref,old_model_ref,args_ref))
            games_played+=1

        #Get remaining  data
        for i in range(workers):
            rewards.append(deepcopy(ray.get(not_ready[i])))
        #del ready
        #del not_ready
        #del dummy_model_ref
        #del game_ref
        print(f"NN evaluation finished. Played out {BATCH_SIZE} games.")
        print(f"Time needed for {BATCH_SIZE} games is: {time.time()-time1:.2f}s")


        #process results, set better model as self.model
        print(f"Draws: {np.sum(np.array(rewards)==0)}")
        print(f"New mode wins: {np.sum(np.array(rewards)==1)}")
        print(f"Old model wins: {np.sum(np.array(rewards)==-1)}")
        eval_outcome = np.sum(rewards)
        if eval_outcome>=0:
            print(f"Saving new model as best")
        else:
            print(f"Saving old model as best")
            self.model = self.old_model
            self.optimizer.load_state_dict(torch.load(os.path.join(self.model_saving_path,"best_optimizer")))
            new_model_better = False
        self.old_model=None

        return new_model_better
    
    def _play_against_random_parallel(self):
        """
        Plays out args["num_games_against_random"] against random agent in parallel
        """
        if self.args["num_games_against_random"]==0:
            return 0
        
        print("Reward evaluation phase starts...")
        time1 = time.time()
        #Utilising resources
        workers = round(mp.cpu_count()*0.5)
        if workers>self.args["num_games_against_random"]:
            workers = self.args["num_games_against_random"]
        print(f"Using {workers} CPU cores...")
        BATCH_SIZE = self.args["num_games_against_random"]


        #Arguments for MCTS used for playing  against random agent
        mcts_args ={
        "MCTS_UCT_c" : 1,
        "MCTS_constraint": "rollouts",
        "MCTS_budget":10,
        "MCTS_dirichlet_alpha": 1,
        "max_moves_num": 60,
        }

        #Run multiple processes and acquire data
        not_ready =[] 
        ready=[]
        rewards= []
        model_ref = ray.put(self.model)
        args_ref = ray.put(mcts_args)
        games_played = 0
        
        while(games_played<BATCH_SIZE):
                if len(not_ready)>=workers:
                    ready, not_ready = ray.wait(not_ready,num_returns=1)
                    rewards.append(deepcopy(ray.get(ready[0])))

                not_ready.append(play_against_random.remote(games_played+1,model_ref,args_ref))
                games_played+=1

            #Get remaining  data
        for i in range(workers):
            rewards.append(deepcopy(ray.get(not_ready[i])))
        #del ready
        #del not_ready
        #del dummy_model_ref
        #del game_ref
        print(f"Reward evaluation finished. Played out {BATCH_SIZE} games.")
        print(f"Time needed for {BATCH_SIZE} games is: {time.time()-time1:.2f}s")
        print(f"Total reward: {np.sum(rewards)}")

        return np.sum(rewards)



#%%
#TESTING OUT

if __name__ == "__main__":
   
    from Model_architecture import HiveAlphaZeroModel
    from utils import Utils

    ray.shutdown()
    ray.init(object_store_memory =400*1000*1000)
    print(ray.available_resources())

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else None)
    model =  HiveAlphaZeroModel()
    #if torch.cuda.device_count() > 1:
    #    print(f"Using{torch.cuda.device_count()} GPUs!")
     #   model = nn.DataParallel(model)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

   
# %% TESTING OUT AlphaZero class(pipeline)

    
    args ={
        "MCTS_UCT_c" : 4,
        "MCTS_constraint": "rollouts",
        "MCTS_budget": 20,
        "MCTS_dirichlet_alpha": 1,
        "num_self_play_iterations":10,
        "max_moves_num_selfplay": 50,
        "num_learning_iterations": 5,
        "num_epochs": 5,
        "batch_size":64,
        "num_evaluation_games":5,
        "num_games_against_random":10
    }
    #Save training hyperparamaters
    model_saving_path = Utils.create_model_folder(args)

    train_pipe = AlphaZero(model,optimizer=optim,cpu=cpu_device,gpu=gpu_device,args=args,model_saving_path=model_saving_path)

    #Start AlphaZero
    start = time.time()
    mem =train_pipe.learn()
    for i in range(10):
        print(i)
    print(f"Time needed to learn: {time.time()-start}")
    #Utils._plot_loss_curve(model_saving_path)
    