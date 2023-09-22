import numpy as np
from copy import deepcopy
import time
import torch
import random
""""
Monte Carlo Tree Search class
"""

class MCTS:
    """
    Monte Carlo Tree Search alghoritm. Uses functon search_tree(self, root)
    to preform MCTS using paramaters specified by initialization of class.

    When performing self-play, user should use truncate_tree(self, old_root)
    in order to truncate tree effectively and use already acquired knwoladge
    in previous search for next search.
    """

    #TODO: **kwargs
    def __init__(self, game_env, UCT_c,constraint = "rollouts", budget = 2000,neural_network=None,UCT_eps=0.25,dirichlet_alpha=None):
        
        self.game_env = game_env #game environment 
        self.game_snapshot = None
        """
        Functions that game_env shoud have in order for compatibility:
        .step(next.action)=> gives next_state, reward, terminated(True/False)
        .legal_actions=> returns legal actions that can be taken from current game state
        """
        self.constraint = constraint #Can be "time" or "rollouts"
        self.budget = budget #time in seconds or number of rollouts
        self.UCT_c = UCT_c #UCT constant
        self.UCT_eps = UCT_eps #TODO: Implement
        self.dirichlet_alpha = dirichlet_alpha #Alpha for dirichlet distribution
        self.neural_net  = neural_network
        self.visited_nodes = [] #nodes visited during one Tree sweep
        self.valid_actions = [0]*3388 #TODO: specific for Chess action space
        self.tau = 1
    
    @torch.no_grad()
    def search_tree(self, root, print_tree = False):
        """
        Main function for MCTS. Does all four stages of MCTS for given root.
        1)Selection is done using tree_policy function
        2)Expansion is done by first adding all possible children nodes to leaf and 
        selecting random children for simulation
        3)Simulation is done upon random children using  random rollouts as default policy
        4)Backpropagation is done using reward_backpropagete function
        Number of search iterations is determined by self.budget
        """
        run_search = True
        search_iteration = 0
        start_time= time.time()
        while run_search:

            self.game_snapshot =self.game_env.getSnapshot() #Make deep copy of game for tree search#TODO: Specific to Hive
            next = root
            self.visited_nodes = []  
            #terminated=False
            game_outcome=0 #game_outcome has following values: 0(running),1(white won), 2(black won), 3(draw)
            state_value=0
            while not game_outcome:
                if len(next.children)==0: 
                    self.visited_nodes.append(next)
                    self.valid_actions =[0]*3388 #TODO:Specific to Hive 
            
                    #expanding node for all possible actions if it isn't expanded and doing rollouts on random child or using NN 
                    self._add_leaf_nodes(next)
                    
                    #Use NN for prediciton if possible
                    if self.neural_net:
                        #Use NN for evaluation
                        tensor_state = torch.tensor(next.state,dtype=torch.float)
                        tensor_state= torch.permute(tensor_state,(2,0,1)).unsqueeze(0)
                        policy, state_value = self.neural_net(tensor_state)
                        policy = torch.nn.functional.softmax(policy, dim=1)
                        
                        policy = policy.numpy().squeeze()
                        state_value = state_value.item()
                        
                        if self.valid_actions: #valid_actions will be None if player needs to pass move
                            #normalize probabilities over possible game states
                            policy*=self.valid_actions
                            policy = policy/np.sum(policy) 
                        
                            #Give corresponding prior prob to every child of node:
                            for child in next.children:
                                child.prior_prob = policy[child.action]
                        else:
                            print("NO VALID ACTION, PASS")

                    else:
                        #Doing random rollouts
                        
                        index = np.random.randint(0,len(next.children))
                        self.visited_nodes.append(next.children[index])
                        next_state, n1,n2,n3=self.game_snapshot.step(next.children[index].action)
                        next.children[index].state = next_state
                        state_value = self._random_rollout(next.children[index])
                    break

                else:
                    #searching through tree
                    self.visited_nodes.append(next)
                    next = self._tree_policy(next)
                    #Perform game step after search and assign state to node selected by tree policy
                    if next.action == -1:
                         #When action is -1 that means player passes move
                         self.game_snapshot.stepPass()
                    else:
                        self.game_snapshot.step(next.action)

                    current_state = self.game_snapshot.getBoardState()
                    game_outcome = self.game_snapshot.getGameStatus() 

                   #current_state,game_outcome,terminated,info=self.game_snapshot.step(next.action)
                   #print(self.game_snapshot.render(mode='unicode'))
                   
                    if (next.state is None):
                        next.state = current_state
                    

            #Backpropagation
            #TODO: Chess environment returns outcome in format:0(white wins), -1(black wins), 0(draw)
            #TODO: Hive environment returns outcome in format:1(white wins), 2(black wins), 3(draw)
            if game_outcome:
                self.visited_nodes.append(next)#If game was terminated we need to add terminal node to visited nodes
                self._reward_backpropagation(state_value,game_outcome)
            else:
                self._reward_backpropagation(state_value,game_outcome)
            search_iteration+=1
            #Check if search exceeds computational budget
            (run_search,comp_time) = self._computational_budget(search_iteration,start_time)
            
        #calculate best child and action probabilities 
        best_child,action_probs = self._find_best_child(root)

        
        
        #Print tree root, and actions 
        if print_tree:
            self._print_tree(root,comp_time,search_iteration)

        return best_child, action_probs
    
    def _find_best_child(self,root):
        """
        Given root node at the end of the search determines best child via AlphaZero criterium if  
        NN is provided, otherwise uses robust child criterium(child with mosts visits is returned). Also  
        calculates probabilities of all actions from root node.
        Returns tuple (best child, action probabilities for child selection).

        """

        best_child = None
        action_probs = [0]*3388 #TODO: Specific to Hive
        #IMPORTANT: We assume that nodes are so arranged that first child node is first possible action and last child node is last possible action!
        #Should be true, check it! #TODO:
        actions = []
        if self.neural_net:
            #Case when player needs to pass(root has only one child which is forced play)
            if len(root.children)==1 and root.children[0].action==-1:
                return root.children[0], action_probs #We will return all zeros as action_probs
            

            for child in root.children:
                action_probs[child.action]=child.visit_cnt**(1/self.tau)
                actions.append(child.visit_cnt**(1/self.tau))
            action_probs= action_probs/np.sum(action_probs)
            actions = actions/np.sum(actions)
            best_child = np.random.choice(root.children,p=actions)


        else:
            for child in root.children:
                action_probs[child.action] =child.visit_cnt
                actions.append(child.visit_cnt)
            most_visits_idx = np.argmax(actions) 
            best_child = root.children[most_visits_idx]
            most_visits_idx = np.argmax(action_probs)
            action_probs = [0]*3388 #TODO: specific to Hive
            action_probs[most_visits_idx] = 1

        return best_child,action_probs  
            
    
    def truncate_tree(self, old_root,new_root,playing_against_random=False,new_state=None):
        """
        Looks for action that was taken by opponent and returns truncated
        tree, so that new root node state is derived by previous action taken by opponent.
        Use this function only when playing against other player. If you want MCTS to play
        against itself don't use this!
        """
        #Truncation when playing against random agent, new root is actually action taken by random agent
        if playing_against_random: #We are waching new_root as action
            if (new_root is not None):
                if not old_root.children:
                    return Node(new_state,None)
                for child in old_root.children:
                    if child.action == new_root:
                        child.state = new_state
                        return child
            else:
                return old_root

        #Truncaton when playing against another MCTS agent
        if new_root.children: 
            if not old_root.children:
                return Node(new_root.state,None)
            for child in old_root.children:
                if child.action==new_root.action:
                    child.state = new_root.state
                    return child
        else:
            return old_root


        
    def _tree_policy(self, node):
        """
        Tree policy for selecting nodes inside of tree using UCT
        """
        if self.neural_net:
            noise = np.random.dirichlet([self.dirichlet_alpha]*len(node.children))
        else:
            noise = None

        UCT_scores = np.empty(len(node.children))
        for i,child in enumerate(node.children):
            if self.neural_net:
                UCT_scores[i] = self._calculate_UCT_score(child,noise[i])
            else:
                UCT_scores[i] = self._calculate_UCT_score(child,None)
        
        argmax = np.argmax(UCT_scores)
        #self.visited_nodes.append(node.children[argmax])
        return node.children[argmax]


    def _calculate_UCT_score(self,node,noise):
        """
        Calculation of Upper Confidance Bound(UCB) for tree node.
        """
        #neural net
        if self.neural_net:
            if node.visit_cnt:
                average_reward = node.total_reward/node.visit_cnt
            else:
                average_reward = 0

            new_prior = (1-self.UCT_eps)*node.prior_prob+self.UCT_eps*noise
            UCT_val = average_reward + new_prior*self.UCT_c*node.parent.visit_cnt**0.5/(1+node.visit_cnt)

        #Random rollouts
        else:
            if node.visit_cnt:
                average_reward = node.total_reward/node.visit_cnt
            else:
                #Node wasn't visted at all, we should visit it!
                return np.inf 
                #average_reward = 0
            UCT_val = average_reward + 2*self.UCT_c*np.sqrt(2*np.log(node.parent.visit_cnt)/node.visit_cnt)
        return UCT_val
    
    def _add_leaf_nodes(self,node):
        """
        Adds leaf nodes to node. Leaf nodes correspond to all possible states that can be achieved from 
        node's state. Also fills out valid_actions vector that is later used for normalizing prior probabilities
        """
        next_possible_actions = self.game_snapshot.getLegalMoves()
        #Case when player can't play, and needs to pass
        if len(next_possible_actions)==0:
            self.valid_actions=None
            child_node = Node(None,node,-1,prior_p=1) #Action -1 defines passing move! It's prior prob is 1!
            node.children.append(child_node)
            return
        #Otherwise add children and fill out valid action vector
        for action in next_possible_actions:
            self.valid_actions[action] =1
            node.children.append(Node(None,node,action))


    
    def _random_rollout(self, node):
        """
        Doing random moves until game terminates on selected node. 
        We are using copy of game for simulations
        """
        #Random moves until terminate state is achieved
        #TODO: Maybe add upper bound on number of moves?
        game_over=0

        while not game_over:
            legal_next_actions = self.game_snapshot.legal_actions
            if len(legal_next_actions)==0:
                #If no legal action is available pass move
                self.game_snapshot.stepPass()
            else:
                #Take random action
                random_action = random.choice(legal_next_actions)
                self.game_snapshot.step(random_action)
            #Get reward
            game_over = self.game_snapshot.getGameStatus()

        return game_over
    
    def _reward_backpropagation(self,state_value,terminated):
        """
        Updates total reward and visit count of all nodes that were
        on the search path. Root included. If terminated is not zero,
        it means we have reached terminal state while searching.
        """
        #Specific to Hive: terminated: 0-running, 1-white won, 2-black won, 3-draw
        if self.neural_net and not terminated:
            reward = state_value*(-1)

        else:
            #Find out who played last move, #TODO:specific to Chess!
            last_player = self.game_snapshot.isWhiteToPlay()#white=>True, black=>False, reffers to player who is currently on move!
            if terminated==3: #game was draw
                reward=0
            elif last_player==True and terminated==1: #white node is last, white won
                reward =-1
            elif last_player==True and terminated==2:#white node is last, black won
                reward = 1
            elif last_player==False and terminated==2:#black node is last, black won
                reward=-1
            elif last_player==False and terminated==1: #black node is last, white won
                reward=1
            else:
                raise ValueError("Invalid player!")
        #Backpropagete
        for node in reversed(self.visited_nodes):
            node.visit_cnt+=1
            node.total_reward+=reward
            reward= reward*(-1)
        #root.visit_cnt+=1
        #root.total_reward = root.total_reward + reward


    def _computational_budget(self,search_iteration, start_time):
        """
        Takes number of search iterations and starting time of search
        and checks if search exceeds computational budget provided by 
        initialization of MCTS class
        """
        if self.constraint=="rollouts":
                if search_iteration==self.budget:
                    return(False, time.time()-start_time)
        elif self.constraint=="time":
            if (time.time()-start_time)>self.budget:
                return(False,time.time()-start_time)
        return (True,0)

    def _print_tree(self, root, delta_time,rollout_number):
        """
        Prints tree of possible actions from root.
        When printing tree. It is clear that from perspective of player
        searching through tree state values are interpreted as:
        [0,1]-In favor of MCTS player winning
        [-1,0] - In favor of opponent winning
        """     
        print("Available actions and their stats")
        print("(total reward, number of visits)->state value %")  
        for child in root.children:
            if child.visit_cnt==0:
                average_reward = 0
            else:
                average_reward = child.total_reward/child.visit_cnt
            print("\t"+"({:.2f}/{})->{:.2f}".format(child.total_reward,child.visit_cnt,average_reward))
        print("Time needed for MCTS: {:.4f}s".format(delta_time))
        print("Number of rollouts: {}".format(rollout_number))
    
    def set_new_model(self,model):
        self.neural_net = model
        




class Node:
    """
    One node of MCTS tree
    """
    index=0 #For debugging
    def __init__(self,state,parent ,action=None,prior_p=1):
        self.state = state
        self.total_reward = 0
        self.visit_cnt = 0
        self.parent = parent
        self.children =[]
        self.prior_prob = prior_p #prior probability of selecting node assigned by parent, needed for AlphaZero UCT
        self.action = action #action needed to select node represented in format of NN policy head output
        self.index = Node.index
        Node.index+=1 #For debugging

    def __repr__(self):
        repr = "Node{},visited:{},action{}".format(self.index,self.visit_cnt,self.action)

        return repr


    
