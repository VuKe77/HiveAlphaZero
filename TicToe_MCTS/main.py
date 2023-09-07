from TicTacToe import TicTacToe
from MCTS import MCTS
from MCTS import Node



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

#Test out MCTS on TicTacToe game

game_env = TicTacToe()
initial_state = game_env.state
root1 = Node(initial_state,None)
root2 = Node(initial_state,None)

game_env.print_board()
MCTS = MCTS(game_env,1.2,"rollouts",1000)
node1 = root1
node2 = root2
player = '1'
#Game where we have one MC agent playing vs another MC agent
while not game_env.outcome:
    if player=='1':
        node1 = MCTS.truncate_tree(node1)
        node1 = MCTS.search_tree(node1,True)
        
        player = '2'
        choice = node1.state
        
    else:
        # human_move = get_human_input()
        # player = '1'
        node2 = MCTS.truncate_tree(node2)
        node2 = MCTS.search_tree(node2,True)
        player = '2'
        choice = node2.state
        
    game_env.step(choice)
    game_env.print_board()

