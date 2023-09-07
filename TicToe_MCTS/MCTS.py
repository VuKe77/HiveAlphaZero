import numpy as np
from copy import deepcopy
import time

class MCTS:
    """
    Monte Carlo Tree Search alghoritm. Uses functon search_tree(self, root)
    to preform MCTS using paramaters specified by initialization of class.

    When performing self-play, user should use truncate_tree(self, old_root)
    in order to truncate tree effectively and use already acquired knwoladge
    in previous search for next search.
    """

    #TODO: **kwargs
    def __init__(self, game_env, UCT_c,constraint = "rollouts", budget = 2000):
        
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
        self.neural_net  = None #TODO: enable neural net prediction
        self.visited_nodes = [] #nodes visited during one Tree sweep
        

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
                    #expanding node for all possible actions if it isn't expanded and doing rollouts on random child
                    self._add_leaf_nodes(next)
                    if len(next.children)==0:
                        rollout_outcome =self.game_env.determine_outcome([next.state])[1]#Beaouse determine_outcome function returns tuple
                        break
                        
                    #Doing random rollouts
                    index = np.random.randint(0,len(next.children))
                    self.visited_nodes.append(next.children[index])
                    rollout_outcome = self._random_rollout(next.children[index])
                    break
                else:
    
                    next = self._tree_policy(next)
                    self.visited_nodes.append(next)

            #Backpropagation
            #TODO: TicTacToe implementation returns outcome in format:"player1_wins, player2_wins, draw"
            self._reward_backpropagation(rollout_outcome,root)
            search_iteration+=1
            #Check if search exceeds computational budget
            (run_search,comp_time) = self._computational_budget(search_iteration,start_time)
            
        #return best child
        best_child = None
        for child in root.children:
            if not best_child:
                best_child = child
            else:
                if child.visit_cnt>best_child.visit_cnt:
                    best_child = child 
        #Print tree root, and actions 
        if print_tree:
            self._print_tree(root,comp_time,search_iteration)

        return best_child
    
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
        UCT_scores = np.empty(len(node.children))
        for i,child in enumerate(node.children):
            UCT_scores[i] = self._calculate_UCT_score(child)
        
        argmax = np.argmax(UCT_scores)
        #self.visited_nodes.append(node.children[argmax])
        return node.children[argmax]


    def _calculate_UCT_score(self,node):
        """
        Calculation of Upper Confidance Bound(UCB) for tree node.
        """
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
        node's state.
        """

        next_possible_states = self.game_env.get_legal_next_states([node.state]) #Needed to be as list becouse of game_env implementation
        for state in next_possible_states:
            node.children.append(Node(state,node))
    
    def _random_rollout(self, node):
        """
        Doing random moves until game terminates on selected node. 
        We are using copy of game for simulations
        """
        #node.visit_cnt+=1
        game_sim = deepcopy(self.game_env)
        #Play states that occurred between the root node and the current node\
        for node in self.visited_nodes:
            game_sim.step(node.state)
        #Random moves until terminate state is achieved
        #TODO: Maybe add upper bound on number of moves?
        while not game_sim.done:
            legal_next_states = game_sim.legal_next_states
            move_idx = np.random.randint(0,len(legal_next_states))
            game_sim.step(legal_next_states[move_idx])

        return game_sim.outcome
    
    def _reward_backpropagation(self,outcome,root):
        """
        Updates total reward and visit count of all nodes that were
        on the search path. Root included.
        """

        root_player =  self.game_env.current_player(root.state)
        #Determine reward
        if root_player=='player1' and outcome=='player1_wins':
            reward =1
        elif root_player=='player1' and outcome =='player2_wins':
            reward =-1
        elif root_player=='player2' and outcome =='player2_wins':
            reward =1
        elif root_player=='player2' and outcome =='player1_wins':
            reward =-1
        else:
            reward = 0
        #Backpropagete
        root.visit_cnt+=1
        root.total_reward = root.total_reward + reward*(-1)
        for node in self.visited_nodes:
            node.visit_cnt+=1
            node.total_reward+=reward
            reward= reward*(-1)

        #self.visited_nodes = []

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
            print("\t"+"({}/{})->{:.2f}".format(child.total_reward,child.visit_cnt,child.total_reward/child.visit_cnt))
        print("Time needed for MCTS: {:.4f}s".format(delta_time))
        print("Number of rollouts: {}".format(rollout_number))
    
    




class Node:
    """
    One node of MCTS tree
    """
    index=0 #For debugging
    def __init__(self,state,parent):
        self.state = state
        self.total_reward = 0
        self.visit_cnt = 0
        self.parent = parent
        self.children =[]
        self.index = Node.index
        Node.index+=1 #For debugging

    def __repr__(self):
        return "Node{},visited:{},reward:{:.2f}".format(self.index,self.visit_cnt,self.total_reward/self.visit_cnt)



    
