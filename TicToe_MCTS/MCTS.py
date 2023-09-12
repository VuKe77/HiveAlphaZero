import numpy as np
from copy import deepcopy
import time
import torch

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
        """
        Functions that game_env shoud have in order for compatibility:
        self.determine_outcome([state])
        self.get_legal_next_states([state])
        self.current_player([state])
        self.step(state)

        Params that game_env should have:
        self.state
        self.done
        self.legal_next_states
        self.outcome
        """
        self.constraint = constraint #Can be "time" or "rollouts"
        self.budget = budget #time in seconds or number of rollouts
        self.UCT_c = UCT_c #UCT constant
        self.UCT_eps = UCT_eps #TODO: Implement
        self.dirichlet_alpha = dirichlet_alpha #Alpha for dirichlet distribution
        self.neural_net  = neural_network
        self.visited_nodes = [] #nodes visited during one Tree sweep
        self.valid_actions = [0,0,0,0,0,0,0,0,0] #TODO: specific for TicTacToe action space
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
            next = root
            self.visited_nodes = []

            while True:
                if len(next.children)==0: 
                    self.visited_nodes.append(next)
                    self.valid_actions =[0,0,0,0,0,0,0,0,0]
                    #expanding node for all possible actions if it isn't expanded and doing rollouts on random child or using NN 
                    self._add_leaf_nodes(next)

                    if len(next.children)==0:
                        #If node wasn't expanded, that means it is terminal. Determine outcome
                        state_value =self.game_env.determine_outcome([next.state])[1]#Because determine_outcome function returns tuple
                        break

                    #Use NN for prediciton if possible
                    if self.neural_net:
                        #Use NN for evaluation
                        tensor_state = torch.tensor(next.state.astype(np.float32)).unsqueeze(0)
                        policy, state_value = self.neural_net(tensor_state)
                        policy = policy.numpy().squeeze()
                        state_value = state_value.item()
                        policy*=self.valid_actions
                        policy = policy/np.sum(policy) #normalize probabilities over possible game states
                    
                        #Give corresponding prior prob to every child of node:
                        for child in next.children:
                            child.prior_prob = policy[child.action]

                    else:
                        #Doing random rollouts
                        
                        index = np.random.randint(0,len(next.children))
                        self.visited_nodes.append(next.children[index])
                        state_value = self._random_rollout(next.children[index])
                    break

                else:
                    #searching through tree
                    self.visited_nodes.append(next)
                    next = self._tree_policy(next)
                    

            #Backpropagation
            #TODO: TicTacToe implementation returns outcome in format:"player1_wins, player2_wins, draw"
            self._reward_backpropagation(state_value,root)
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
        action_probs = [0]*9 #TODO: Specific to TicTacToe action size of 9
        #IMPORTANT: We assume that nodes are so arranged that first child node is first possible action and last child node is last possible action!
        #Should be true, check it! #TODO:
        actions = []
        if self.neural_net:
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
            action_probs = [0]*9
            action_probs[most_visits_idx] = 1

        return best_child,action_probs  
            
    
    def truncate_tree(self, old_root):
        """
        Looks for action that was taken by opponent and returns truncated
        tree, so that new root node state is derived by previous action taken by opponent.
        """
        new_state = self.game_env.state
        for child in old_root.children:
            if (child.state == new_state).all():
                return child
        return Node(new_state,old_root)

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


        next_possible_states = self.game_env.get_legal_next_states([node.state]) #Needed to be as list becouse of game_env implementation of TicTacToe
        for state in next_possible_states:
            action = self._action_mapping(node.state,state)
            self.valid_actions[action] =1
            node.children.append(Node(state,node,action))

    def _action_mapping(self,current_state, next_state):
        """
        This is action mapping specifically made for TicTacToe game, where action space is  
        in format: [[3x3 matrix representing x's], [3x3 matrix representing o's], [3x3 matrix with 1's or 0's(dependig on player)]].
        Needed in order for us to know which prior probability to assign to which node(i.e masking NN policy head output).
        """
        current_state = np.array(current_state)#state of node that we are currently in
        next_state = np.array(next_state) #state of node we want to go in
        player = current_state[2][0][0]
        if player ==0:
            old_state = current_state[0].flatten()
            new_state = next_state[0].flatten()
        elif player==1:
            old_state = current_state[1].flatten()
            new_state = next_state[1].flatten()

        else: 
            raise ValueError("Player Invalid")
        
        action = np.where(old_state!=new_state)[0][0]
        return action
    
    
    def _random_rollout(self, node):
        """
        Doing random moves until game terminates on selected node. 
        We are using copy of game for simulations
        """
        #node.visit_cnt+=1
        game_sim = deepcopy(self.game_env)
        #Play states that occurred between the root node and the current node\
        for i in range(1,len(self.visited_nodes)): #skip root
            game_sim.step(self.visited_nodes[i].state)
        #Random moves until terminate state is achieved
        #TODO: Maybe add upper bound on number of moves?
        while not game_sim.done:
            legal_next_states = game_sim.legal_next_states
            move_idx = np.random.randint(0,len(legal_next_states))
            game_sim.step(legal_next_states[move_idx])

        return game_sim.outcome
    
    def _reward_backpropagation(self,state_value,root):
        """
        Updates total reward and visit count of all nodes that were
        on the search path. Root included. If state_value is string
        it means we have reached terminal state while searching and it returns "player1_wins", "player2_wins".
        If state_value is float we have used NN to get value approximation.
        """
        if self.neural_net and (not isinstance(state_value,str)):
            reward = state_value

        else:
            #Find out who played last move, #TODO:specific to TicTacToe!
            last_player = self.visited_nodes[-1].state[2][0][0]#player1=>0, player2=>1

            #root_player =  self.game_env.current_player(root.state)
            #Determine reward
            if last_player==0 and state_value=='player1_wins':
                reward =-1 #Because if terminated node player was p1, action was taken by player2 and it was bad!
            elif last_player==0 and state_value =='player2_wins':
                reward =1
            elif last_player==1 and state_value =='player2_wins':
                reward =-1
            elif last_player==1 and state_value =='player1_wins':
                reward =1
            else:
                reward = 0
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
        repr = "Node{},visited:{}".format(self.index,self.visit_cnt)

        return repr


    
