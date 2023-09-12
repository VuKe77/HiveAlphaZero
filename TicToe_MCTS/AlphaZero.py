#%%
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

#Ignore VisiblDeprecationWarning, annoying(comes up when we want to make arrays of action probs in training func)
np.warnings.filterwarnings(
    'ignore',
    category=np.VisibleDeprecationWarning
)
#from tqdm.notebook import trange




class ResNet(nn.Module):
    """
    Neural Net that is used for learning in Alpha Zero algorithm
    """

    def __init__(self, game, num_resBlocks, num_hidden):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(3,num_hidden,kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*3*3, 9),# 3*3 is size of board, action size is 9
            nn.Softmax(dim=1)
        )
        self.valeHead = nn.Sequential(
            nn.Conv2d(num_hidden,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*3*3, 1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valeHead(x)
        return policy, value

class ResBlock(nn.Module):
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
    def __init__(self,model, optimizer, game, args):
        """
        Training pipeline class for AlphaZero alghorihm.
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
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

        ''' #MCTS_UCT_c, MCTS_constraint, MCTS_budget,num_self_play_iterations, max_moves_num
        self.mcts = MCTS(self.game,self.args["MCTS_UCT_c"],self.args["MCTS_constraint"],
                         self.args["MCTS_budget"], self.model, dirichlet_alpha=self.args["MCTS_dirichlet_alpha"])

    def _self_play(self,verbose=False):
        """
        Performs number of self-play games defined in args("num_self_play_iterations"). Every self-play game has early  
        termination if number of moves exceeds "max_moves_num". Returns vector containing every game trajectory, where  
        every move is in format [state of game, posterior action probs, reward] 
        """
        starting_time = time.time()
        self_play_buffer = []
        for i in range(self.args["num_self_play_iterations"]):
            #Prepare loop for new gameplay
            self.game.reset()
            root = Node(self.game.state,None)
            turn_cnt =0
            one_game_data = []
            game_start_time = time.time()
            while not self.game.outcome:
                #Playout the game and save game trajectories
                
                if turn_cnt==self.args["max_moves_num"]:
                    break #early termination   
                root = self.mcts.truncate_tree(old_root=root)
                old_state = root.state
                player = root.state[2][0][0] #TODO: Specific for TicTacToe, rework for another game
                root, action_prob = self.mcts.search_tree(root)
                self.game.step(root.state)

                one_game_data.append([old_state,action_prob,player]) #We are currently saving player in place of reward, when game is over we replace it with reward
                turn_cnt+=1
            #When game is over add reward to one_game_data instead of player => P1 win(+1), P2 win(-1), draw(+0)
            #TODO: specific for TicTacToe
            if self.game.outcome:
                for data in one_game_data:
                    if self.game.outcome=="player1_wins":
                        if data[2]==0:
                            data[2]=1
                        else:
                            data[2]=-1
                    elif self.game.outcome=="player2_wins":
                        if data[2]==1:
                            data[2]=1
                        else:
                            data[2]=-1
                    elif self.game.outcome=="draw":
                        data[2]=0
                    else:
                        raise ValueError("Uncategorized winner!")

            self_play_buffer+=one_game_data #Add game trajectory to memory
                
            
        if verbose:
            delta_time = time.time() - starting_time
            print(f"Self-play number of games performed: {self.args['num_self_play_iterations']}.")
            print(f"Time needed: {delta_time:.2f}s ")
            
        
        return self_play_buffer
            
    def _train(self, memory):
        """
        Here is NN trained on memory data in batches of size args['batch_size'].
        Memory data is shuffled first. Returns average loss of training on memory
        as 2D vector(policy loss and value loss)

        """

        cumalative_loss=[[],[]] #Used for plotting loss curve, first array is policy lost, second is value loss

        #shuffle training data
        #training_data = memory
        random.shuffle(memory)
        for batch_idx in range(0,len(memory),self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory),batch_idx+self.args['batch_size'])]
            states, policy_targets, value_targets = zip(*sample)
            states, policy_targets, value_targets = np.array(states), np.array(policy_targets), np.array(value_targets).reshape(-1,1) #TODO:need this?
            
            #Turn into tensors, prepare for NN
            states = torch.tensor(states,dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets,dtype=torch.float32)
            value_targets = torch.tensor(value_targets,dtype=torch.float32)

            out_policy, out_value = self.model(states)

            policy_loss = F.cross_entropy(out_policy,policy_targets)#Ne slaze se sa njegovim, on nema softmax??
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss+value_loss
            cumalative_loss[0].append(policy_loss.detach().numpy())  #used for plotting loss curve
            cumalative_loss[1].append(value_loss.detach().numpy())  #used for plotting loss curve

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
        this process is repeated is defined by 'num_learning_iterations'. If loss_verbose is True,
        and the end of every learning epoch loss curve is plotted
        """
        average_policy_loss_vector = [] #Policy loss is accumulated during training phase and than averaged and appended to this vector
        average_value_loss_vector = [] #Value loss is accumulated during training phase and than averaged and appended to this vector
        reward_history=[] #After every learning iterations games are played 10 against random agent, and final reward is appended to this vector
        for iter in tqdm(range(self.args['num_learning_iterations'])):
            torch.save(self.model.state_dict(),f"best_model")
            torch.save(self.optimizer.state_dict(), f"best_optimizer")
            memory = [] #vector, whos elements are self-played games#TODO:Maybe put upper bound on memory limit

            #Perform self-play to gather data for training 
            self.model.eval()
            memory+= self._self_play()
            
            #Train model on gathered data
            self.model.train()
            for epoch in range(self.args["num_epochs"]):
               average_loss = self._train(memory)
               average_policy_loss_vector.append(average_loss[0])
               average_value_loss_vector.append(average_loss[1])
            #Play with old model to evaluate NN
            self._evaluate_NN()

            #Play 10 games against random agent in order to evaluate network later
            reward = self._play_against_random()
            reward_history.append(reward) 

        #Save best model, save loss values during training, save rewards, and plot loss and reward.   
        torch.save(self.model.state_dict(),f"best_model")
        torch.save(self.optimizer.state_dict(), f"best_optimizer")
        with open('loss_data.npy','wb') as f:
            np.save(f,average_policy_loss_vector)
            np.save(f,average_value_loss_vector)
            np.save(f,reward_history)
        self._plot_loss_curve(average_policy_loss_vector,average_value_loss_vector,reward_history)


    def _plot_loss_curve(self,policy_loss_vector, value_loss_vector,reward=None):
        """
        Used for potting loss on graph. If reward is provided than plots reward also.
        """
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(1,len(policy_loss_vector)+1),policy_loss_vector)
        plt.ylabel("Policy loss")
        plt.xlabel("epochs of training")
        plt.subplot(2,1,2)
        plt.plot(range(1,len(policy_loss_vector)+1),value_loss_vector)
        plt.ylabel("value_loss")
        plt.xlabel("epochs of training")
        plt.show()
        #Plot reward also
        if reward:
            plt.figure()
            plt.plot(range(1,len(reward)+1),reward)
            plt.ylabel("Reward")
            plt.xlabel("Number of learning iterations")
            plt.show()

        
    def _evaluate_NN(self):
        """
        Previouse and new NN play specific number of games against each other. If new
        NN outperforms old one by 55%, we take new NN, otherwise old NN is perserved.
        Uses MCTS with 200 rollouts. Every game first player is changed to allow both NN
        play first move equal amount of times.
        """
        #Load old model, and set boths models in evaluation mode
        if self.args["num_evaluation_games"]==0:
            return

        old_model = deepcopy(model)
        old_model.load_state_dict(torch.load("best_model"))
        old_model.eval()
        self.model.eval()
        mcts = MCTS(self.game,1.41,"rollouts",budget = 200,neural_network=model,dirichlet_alpha=1) #MCTS used for evaluation games
        old_model_wins =0
        new_model_wins=0
        draws=0
        print("EVALUATION PHASE STARTS!")
        #perform tournament
        for game_iter in range(self.args["num_evaluation_games"]):
            #Reset game, make nodes for MCTS and set starting_player
            #print(f"Playing out game {game_iter+1}") DEBUG
            self.game.reset()
            node1 = Node(game.state,None) #MCTS for old model
            node2 = Node(game.state,None) #MCTS for new model
            if game_iter%2==0:
                current_player = '1'
            else:
                current_player = '2'
            #Playout games
            while not self.game.outcome:
                if current_player=='1':
                    mcts.set_new_model(old_model) #set model for MCTs
                    node1 = mcts.truncate_tree(node1)
                    node1,a1 = mcts.search_tree(node1)
                    current_player = '2'
                    choice = node1.state
                    
                elif current_player =='2':
                    mcts.set_new_model(self.model) #set old model as MCTS NN to prepare for next player
                    node2 = mcts.truncate_tree(node2)
                    node2,a2 = mcts.search_tree(node2)
                    current_player = '1'
                    choice = node2.state
                   
                self.game.step(choice)
                #self.game.print_board()  #DEBUG

            #Check what NN won the game:
            if self.game.outcome =="player1_wins":
                if game_iter%2 ==0: #this means player1 was old NN
                    old_model_wins+=1
                else:
                    new_model_wins+=1

            elif self.game.outcome == "player2_wins":
                if game_iter%2 ==0: #this means player1 was old NN
                    new_model_wins+=1
                else:
                    old_model_wins+=1
            elif self.game.outcome == "draw":
                draws+=1
            else:
                raise ValueError("Invalid outcome!")
            
        #Check stats and set new best network
        if new_model_wins+old_model_wins==0:
            win_per=0
        else:
            win_per = new_model_wins/(new_model_wins+old_model_wins)
        if win_per>0.55:
            print(f"New model is better,by winning {100*win_per:.2f}% of games")
            
        else:
            print(f"Old model is better, by winning {100*(1-win_per):.2f}% of games")
            self.model = old_model
            self.optimizer.load_state_dict(torch.load("best_optimizer"))
        print(f"Draws:{draws}")
            
        return
    
    def _play_against_random(self):
        """
        NN plays against random agent to see how much reward will be acquierd. Ten  
        games are played. Both NN and random agent have their turns playing firs.
        FOr every won game reward is +1, for every lost game reward is -1.  
        Returns reward after ten games
        """
        reward=0
        
        mcts_dummy = MCTS(self.game,1.41,"rollouts",budget = 200,neural_network=None,dirichlet_alpha=1) #We need this just for action mapping function
        for i in range(10):
            if i%2==0:
                player='1'
                first_player = "NN"
            else:
                player='2'
                first_player="RANDOM"
            self.game.reset()
            played_moves = []
            #self.game.print_board()
            while not self.game.outcome:
                if player=='1':
                    tensor_state = torch.tensor(self.game.state.astype(np.float32)).unsqueeze(0)
                    policy, value =model(tensor_state)
                    value = value.item()
                    policy = policy.squeeze(0).detach().numpy()
                    target_index = np.argmax(policy)

                    while target_index in played_moves:
                        policy[target_index]=0
                        target_index=np.argmax(policy)
                    if self.game.current_player(self.game.state) == 'player1':
                        next_state = self.game.state
                        next_state[0][target_index//3][target_index%3]=1
                        next_state[2] = np.ones((3,3))
                    elif self.game.current_player(self.game.state) == 'player2':
                        next_state = self.game.state
                        next_state[1][target_index//3][target_index%3]=1
                        next_state[2] = np.zeros((3,3))

                    played_moves.append(target_index)
                    self.game.step(next_state)
                    player='2'

                elif player=='2':
                    #Random move
                    available_moves = game.legal_next_states
                    next_state = random.choice(available_moves)
                    played_moves.append(mcts_dummy._action_mapping(game.state,next_state))
                    
                    game.step(next_state)
                    player='1'
                #game.print_board()
            
            if game.outcome=='player1_wins':
                if first_player=="NN":
                    reward+=1
                else:
                    reward-=1
            elif game.outcome =='player2_wins':
                if first_player=="RANDOM":
                    reward+=1
                else:
                    reward-=1
        return reward

        





#%%
#TESTING OUT

if __name__ == "__main__":
    from TicTacToe import TicTacToe
    import matplotlib.pyplot as plt
    game = TicTacToe()
    model = ResNet(game,4,64)
    optim = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)

    #model.load_state_dict(torch.load('best_model'))
    #optim = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
    #optim.load_state_dict(torch.load('best_optimizer'))
   

    #%% TESTING ResNet
    '''
    rand_idx = np.random.randint(0,len(game.legal_next_states))
    next_state = game.legal_next_states[rand_idx]
    tensor_state = torch.tensor(next_state.astype(np.float32)).unsqueeze(0)
    model.eval()
    policy, value =model(tensor_state)
    value = value.item()
    policy = policy.squeeze(0).detach().numpy()
    plt.bar(range(9),policy)
    '''
# %% TESTING OUT AlphaZero class(pipeline)
#MCTS_UCT_c, MCTS_constraint, MCTS_budget,num_self_play_iterations, max_moves_num
    
    args ={
        "MCTS_UCT_c" : 4,
        "MCTS_constraint": "rollouts",
        "MCTS_budget": 200,
        "MCTS_dirichlet_alpha": 1,
        "num_self_play_iterations":4,
        "max_moves_num": 10,
        "num_learning_iterations": 5,
        "num_epochs": 3,
        "batch_size":64,
        "num_evaluation_games":0
    }
    train_pipe = AlphaZero(model,optimizer=optim,game=game,args=args)
    start = time.time()
    mem =train_pipe.learn()
    print(f"Time needed to learn: {time.time()-start}")
    
    



# %% Visualize trained nn through self-play game
    played_moves = []
    plt.figure()
    game.reset()
    game.print_board()
    model.load_state_dict(torch.load('best_model'))
    model.eval()

    while not game.outcome:
        #for i in range(5):
        plt.clf()
   
         
        tensor_state = torch.tensor(game.state.astype(np.float32)).unsqueeze(0)
        policy, value =model(tensor_state)
        value = value.item()
        policy = policy.squeeze(0).detach().numpy()
       
        plt.bar(range(9),policy)
        plt.title(value)
        plt.show()

        target_index = np.argmax(policy)
        while target_index in played_moves:
            policy[target_index]=0
            target_index=np.argmax(policy)

        if game.current_player(game.state) == 'player1':
            next_state = game.state
            next_state[0][target_index//3][target_index%3]=1
            next_state[2] = np.ones((3,3))
        elif game.current_player(game.state) == 'player2':
            next_state = game.state
            next_state[1][target_index//3][target_index%3]=1
            next_state[2] = np.zeros((3,3))
        played_moves.append(target_index)
        #print(next_state)
        game.step(next_state)
        game.print_board()
