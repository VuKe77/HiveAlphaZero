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



class DummyNN(nn.Module):
    def __init__(self,num_resBlocks,num_hidden):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(119,num_hidden,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [DummyResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden,out_channels=num_hidden,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_hidden*8*8,4672)
            
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*8*8,1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.startBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
class DummyResBlock(nn.Module):
    """
    One block of residual layers
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
    def forward(self, x):

        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AlphaZero:
    def __init__(self,model, optimizer, game,device, args,model_saving_path):
        """
        Training pipeline class for AlphaZero alghorihm.
        """
        self.device = device 
        self.model = model
        self.old_model = None 
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.model_saving_path = model_saving_path #Path for saving our model
        ''' ARGS:
        -MCTS_UCT_c: MCTS constant for UCB
        -MCTS_constraint: training constraint, can be "rollouts" or "time"
        -MCTS_budget: budget for every MCTS iteration, if constraint is rollouts: number of rollouts
        if constraint is time: time in seconds
        -MCTS_dirichlet_alpha: alpha used in dirichlet distribution
        -num_self_play_iterations:  number of self-play games performed in one cycle of learning 
        -max_moves_num: constraint of maximum number of moves that can be performed in one self-play game. If max_moves_num
        is achieved game is declared draw!
        -num_learning_iterations: number of learning iterations. One learning iteration consists of:
        self-play, training on self-play data, NN evaluation
        -num_epochs: number of swipes through data acquiered by self-pay in training phase
        -batch_size: size of one batch used for training data with mini-bathces
        -num_evaluation_games: number of evaluation games played in evaluation phase
        -num_games_against_random: number of games played against random agent in order to check reward

        ''' 
    def _parallel_self_play(self,verbose=False):
        """
        Performs number of self-play games defined in args("num_self_play_iterations"). Every self-play game has early  
        termination if number of moves exceeds "max_moves_num". Returns vector containing every game trajectory, where  
        every move is in format [state of game, posterior action probs, reward]. Parallelized version using Pool()
        """
        self_play_indices = [i for i in range(self.args["num_self_play_iterations"])]
        workers = mp.cpu_count()
        print(f"Using {workers//2} workers")

        self_play_pool = Pool(workers//2)
        self_play_buffer = self_play_pool.map(self._self_play,self_play_indices)
        self_play_pool.close()
        self_play_pool.join()

        merged_data=[]     
        #merge games together
        for game in self_play_buffer:
            merged_data+=game
        return merged_data

    def _self_play(self,index):
        """
        Performs self-play games. Every self-play game has early  
        termination if number of moves exceeds "max_moves_num". Returns vector containing  game trajectory, where  
        every move is in format [state of game, posterior action probs, reward] 
        """
        print(f"Game:{index}")
        #Prepare loop for new gameplay
        game_copy = deepcopy(self.game)
        initial_state = game_copy.reset()
        mcts = MCTS(game_copy,self.args["MCTS_UCT_c"],self.args["MCTS_constraint"],
                         self.args["MCTS_budget"], self.model, dirichlet_alpha=self.args["MCTS_dirichlet_alpha"])
        root = Node(initial_state,None)
        turn_cnt =0
        one_game_data = []
        terminated = False
        while not terminated:
            #Playout the game and save game trajectories
            if turn_cnt==self.args["max_moves_num"]:
                terminated = True
                break #early termination 

            old_state = root.state
            player = root.state[:,:,112][0][0] #TODO: Specific for Chess, rework for another game
            root, action_prob = mcts.search_tree(root)
            new_state,game_outcome,terminated,info = game_copy.step(root.action)

            one_game_data.append([old_state,action_prob,player]) #We are currently saving player in place of reward, when game is over we replace it with reward
            turn_cnt+=1

       
        #When game is over add reward to one_game_data instead of player => P1 win(+1), P2 win(-1), draw(+0)
        #TODO: specific for Chess
        for data in one_game_data:
            if game_outcome==0: #draw
                data[2]=0
            elif game_outcome==1: #white player wins
                if data[2]==1: #white was on move
                    data[2]=1
                else: #black was on move
                    data[2]=-1
            elif game_outcome ==-1: #black player wins
                if data[2]==1:#white was on move
                    data[2]=-1 
                else:#black was on move
                    data[2]=-1         
            else:
                raise ValueError("Uncategorized winner!")
                
        return one_game_data
            
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
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            out_policy, out_value = self.model(states)

            policy_loss = F.cross_entropy(out_policy,policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss+value_loss
            cumalative_loss[0].append(policy_loss.to(cpu_dev).detach().numpy())  #used for plotting loss curve
            cumalative_loss[1].append(value_loss.to(cpu_dev).detach().numpy())  #used for plotting loss curve

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
        self.model.to(cpu_dev) #Before start of the loop move model to CPU
        for iter in tqdm(range(self.args['num_learning_iterations'])):
            torch.save(self.model.state_dict(),os.path.join(self.model_saving_path,"best_model"))
            torch.save(self.optimizer.state_dict(),os.path.join(self.model_saving_path,"best_optimizer"))
            memory = [] #vector, whos elements are self-played games #TODO:Maybe put upper bound on memory limit

            #Perform self-play to gather data for training 
            self.model.eval()
            time1 = time.time()
            memory+= self._parallel_self_play()
            print(f"Self play time for {self.args['num_self_play_iterations']} games is: {time.time()-time1:.2f}s")
            
            #Train model on gathered data
            self.model.train()
            self.model.to(self.device)#Move model to GPU for training
            print("Learning phase starts...")
            for epoch in range(self.args["num_epochs"]):
               average_loss = self._train(memory)
               average_policy_loss_vector.append(average_loss[0])
               average_value_loss_vector.append(average_loss[1])

            #Play with old model to evaluate NN
            self.model.to(cpu_dev) #Move model to CPU for evaluation
            if self._evaluate_NN_parallel():
                #If new model is better we will playout games against random to see who's better
                #Play  games against random agent in order to evaluate network later
                reward = self._play_against_random_parallel()
                reward_history.append(reward) 


            #save loss values and rewards during training   
            with open(os.path.join(self.model_saving_path,'loss_data.npz'),'wb') as f:
                np.savez(f,policy_loss=average_policy_loss_vector,value_loss = average_value_loss_vector, reward = reward_history)


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
        #Load old model, prepare for evaluation
        self.old_model =  deepcopy(self.model) #We copy model outside of class, just to get skeleton
        self.old_model.load_state_dict(torch.load(os.path.join(self.model_saving_path,"best_model")))
        self.old_model.eval()
        self.model.eval()

        #Create pool for multiprocessing
        evaluate_play_indices = [i for i in range(self.args["num_games_against_random"])]
        workers = mp.cpu_count()//4
        if len(evaluate_play_indices)<workers:
            workers = len(evaluate_play_indices)
        print(f"Evaluation phase starts, playing {self.args['num_evaluation_games']} against old model...")
        print(f"Using {workers} workers")
        evaluate_play_pool = Pool(workers)
        #Playout games and close pool
        game_outcomes = evaluate_play_pool.map(self._evaluate_NN,evaluate_play_indices)
        evaluate_play_pool.close()
        evaluate_play_pool.join()

        #process results, set better model as self.model
        game_outcome = np.sum(game_outcomes)
        print(f"Draws: {len(game_outcomes)-np.count_nonzero(game_outcomes)}")
        if game_outcome>=0:
            print(f"New model is better,by winning {game_outcome}/{len(game_outcomes)} games")
        else:
            print(f"Old model is better,by winning {-game_outcome}/{len(game_outcomes)} games")
            self.model = self.old_model
            self.optimizer.load_state_dict(torch.load("best_optimizer"))
            new_model_better = False
        self.old_model=None
        return new_model_better

    
    def _evaluate_NN(self,index):
        """
        Plays out game with new model and old model, to see which is better
        """
        #TODO: Set parameters for self play evaluation MCTS
        print(index)
        game_copy = deepcopy(self.game)
        mcts = MCTS(game_copy,1.41,"rollouts",budget = 20,neural_network=self.model,dirichlet_alpha=1) #MCTS used for evaluation games      
        max_moves=40
        
        #perform tournament
       
        #Reset game, make nodes for MCTS and set starting_player
        #print(f"Playing out game {game_iter+1}") DEBUG
        move_cnt=0
        initial_state = game_copy.reset()
        node1 = Node(initial_state,None) #MCTS for old model
        node2 = Node(initial_state,None) #MCTS for new model
        if index%2==0:
            current_player = '1'
        else:
            current_player = '2'
        #Playout games
        terminated = False
        while not terminated:
            if move_cnt==max_moves:
                break
            if current_player=='1':
                mcts.set_new_model(self.old_model) #set model for MCTs
                node1 = mcts.truncate_tree(node1,node2)
                node1,a1 = mcts.search_tree(node1)
                current_player = '2'
                action_taken = node1.action
                
            elif current_player =='2':
                mcts.set_new_model(self.model) #set old model as MCTS NN to prepare for next player
                node2 = mcts.truncate_tree(node2,node1)
                node2,a2 = mcts.search_tree(node2)
                current_player = '1'
                action_taken = node2.action

            new_state,game_outcome,terminated,info = game_copy.step(action_taken)
            move_cnt+=1
            #print(self.game.render(mode='unicode'))  #DEBUG
        #Check what NN won the game:
        if game_outcome ==1: #white wins
            if index%2 ==0: #old model is white 
                return -1
            else:
                return 1

        elif game_outcome == -1: #black wins
            if index%2 ==0: #old model white
                return 1
            else:
                return -1
        elif game_outcome == 0: #draw
            return 0
        else:
            raise ValueError("Invalid outcome!")
            

    
    def _play_against_random_parallel(self):
        """
        Plays out args["num_games_against_random"] against random agent in parallel using
        self._play_against_random() function
        """
        if self.args["num_games_against_random"]==0:
            return 0
        random_play_indices = [i for i in range(self.args["num_games_against_random"])]
        workers = mp.cpu_count()//4   
        if len(random_play_indices)<workers:
            workers = len(random_play_indices)
        print(f"Playing {args['num_games_against_random']} games against random agent for reward evaluation...")
        print(f"Using {workers} workers")

        random_play_pool = Pool(workers)
        rewards = random_play_pool.map(self._play_against_random,random_play_indices)
        random_play_pool.close()
        random_play_pool.join()
        rewards = np.sum(rewards)
        return rewards

    
    def _play_against_random(self,index):
        """
        NN plays against random agent Both NN and random agent have their turns playing firs.
        For every  game won by NN reward is +1, for every lost game reward is -1.  
        Returns reward,
        """
        print(index)
        game_copy = deepcopy(self.game)
        reward=0
        if index%2==0:
            player='1'
            #first_player = "NN"
        else:
            player='2'
            #first_player="RANDOM"
        current_state = game_copy.reset()
        terminated= False
        while not terminated:
            if player=='1':
                #use NN to get next action
                tensor_state = torch.tensor(current_state.astype(np.float32))
                tensor_state = torch.permute(tensor_state,(2,0,1)).unsqueeze(0)
                policy, value =self.model(tensor_state)
                value = value.item()
                policy = policy.squeeze(0).detach().numpy()

                #Mask out invalid actions and normilize policy vector probabilities
                valid_actions = [0]*4672
                for action in game_copy.legal_actions:
                    valid_actions[action]=1
                policy*=valid_actions
                policy=policy/(np.sum(policy)+1e-10)

                #find highest prob action
                action_taken = np.argmax(policy)
                #Sometimes it can happen that masked policy had 0 values, so wrong action would be taken
                if action_taken not in game_copy.legal_actions:
                    #Radnom move
                    action_taken = random.choice(game_copy.legal_actions)
                player='2'

            elif player=='2':
                #Random move
                action_taken = random.choice(game_copy.legal_actions)
                player='1'

            current_state,game_outcome,terminated,info = game_copy.step(action_taken)
            #print(self.game.render(mode='unicode')) #DEBUG

        #determine reward
        if game_outcome==1:#white wins
            if index%2==0: #White player is NN
                reward+=1
            else:
                reward-=1
        elif game_outcome ==-1:#black wins
            if index%2==1:#white player is random
                reward+=1
            else:
                reward-=1
        return reward

#%%
#TESTING OUT

if __name__ == "__main__":
    import gym
    import gym_chess
    from utils import Utils

    ray.shutdown()
    ray.init()
    game = gym.make("ChessAlphaZero-v0")

    cpu_dev = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DummyNN(10,64)
    if torch.cuda.device_count() > 1:
        print(f"Using{torch.cuda.device_cout()} GPUs!")
        model = nn.DataParallel(model)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

   
# %% TESTING OUT AlphaZero class(pipeline)
#MCTS_UCT_c, MCTS_constraint, MCTS_budget,num_self_play_iterations, max_moves_num
    
    args ={
        "MCTS_UCT_c" : 4,
        "MCTS_constraint": "rollouts",
        "MCTS_budget": 100,
        "MCTS_dirichlet_alpha": 1,
        "num_self_play_iterations":28,
        "max_moves_num": 65,
        "num_learning_iterations": 3,
        "num_epochs": 5,
        "batch_size":64,
        "num_evaluation_games":0,
        "num_games_against_random":5
    }
    #Save training hyperparamaters
    model_saving_path = Utils.create_model_folder(args)

    train_pipe = AlphaZero(model,optimizer=optim,game=game,device=device,args=args,model_saving_path=model_saving_path)

    #Start AlphaZero
    start = time.time()
    mem =train_pipe.learn()
    print(f"Time needed to learn: {time.time()-start}")
    #Utils._plot_loss_curve(model_saving_path)
    