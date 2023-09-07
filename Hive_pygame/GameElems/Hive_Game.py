import pygame as pg
from GameElems.Board import Board
from GameElems.Pieces.Free     import Free
from GameElems.Pieces.Piece import Piece
from GameElems.constants import SCREENHEIGHT, SCREENWIDTH, BACKGROUND
"""
Look in Board class for explanation of hex_norm
"""
n = [1,-1,0]
nw = [0,-1,1]
sw = [-1,0,1]
s = [-1,1,0]
se = [0,1,-1]
ne = [1,0,-1]
hex_norm = [n,nw,sw,s,se,ne]


class Game:
    def __init__(self, win):
        self.win = win
        self.restart_game()
    

    def update(self):
        """
        Updates game screen, needed for visualization
        """
        
        if self.game_over:
            font = pg.font.Font(None,64)
            text = font.render("Game over! Press 'R' to restart ",True,(255,255,255))
            textpos = text.get_rect(centerx = SCREENWIDTH/2,centery = SCREENHEIGHT/2)
            self.win.blit(text,textpos)

        else:
            self.board.draw(self.win)

        pg.display.update()

    def restart_game(self):
        """
        Initializes game to starting position
        """
        self.selected = None
        self.board = Board()
        self.turn = "w"
        self.valid_moves = [] #Valid moves for this turn, with respect to selected piece
        self.white_queen = self.board.board["white_pieces_out"][0]
        self.black_queen = self.board.board["black_pieces_out"][0]
        self.game_over = False
        self.turn_number = 0

    


    def select_piece(self, mouse_pos):
        """
        Selects piece when mouse press is released (mouse_pos). If piece is already selected, and
        valid Free tile is selected, piece is moved to Free Tile. If same piece is selected twice
        unselects selected piece
        """

        flag = False #flag for optimization
        board = self.board.get_board()
        #if no piece is selected, select piece if mouse pressed it.
        if not self.selected:
            for key,pieces in board.items():
                #check which piece is selected and select it
                if (self.turn == "w" and (key=="white_pieces_in" or key == "white_pieces_out")) or\
                    (self.turn == "b" and (key=="black_pieces_in" or key == "black_pieces_out")):        
                    for piece in pieces:
                        if piece.piece_frame_rect.collidepoint(mouse_pos):
                            flag = True

                            #Game rule: can't move pieces in play if queen is not on board
                            if piece.coordinates != [-1,-1,-1]:
                                if piece.player =="w" and not(self.white_queen.in_play):
                                    print("Can't move piece, until queen is on board!")
                                    return
                                elif piece.player =="b" and not(self.black_queen.in_play):
                                    print("Can't move piece, until queen is on board!")
                                    return
                            #Game rule: must insert queen until fourth turn
                            if self.turn_number>5:
                                if self.turn=="w" and piece!=self.white_queen and not(self.white_queen.in_play):
                                    print("You must move your queen in the game!")
                                    return
                                elif self.turn =="b" and piece!=self.black_queen and not(self.black_queen.in_play):
                                    print("You must move your queen in the game!")
                                    return

                            #we can't select piece under beetle
                            if piece.piece_on_top: 
                                continue

                            #first check if moving piece would break hive, don't select it if True
                            if self.board.check_hive_connectivity(piece):
                                print("Can't move piece, it will break hive")
                            #select piece and get valid moves for selected piece
                            else:
                                self.selected = piece
                                self.valid_moves= piece.valid_moves(self.board.get_board(),self.turn,self.turn_number)
                                piece.select()
                                
                    if flag:
                        break
            


        else:
            #unselect selected piece
            if self.selected.piece_frame_rect.collidepoint(mouse_pos):
                #TODO:Visualize
                #when tile is unselected change frames back to normal
                for tile in self.valid_moves:
                    tile.unset_valid_frame()
                self.selected.unselect()
                self.selected = None
              

            #move piece to free tile
            dst_piece=None
            
            for piece in self.valid_moves:
                if piece.piece_frame_rect.collidepoint(mouse_pos):
                    dst_piece = piece
            if dst_piece: #for solving bug, when mouse is both on free tile and piece
                if self.selected:
                     #TODO:Visualize
                    for tile in self.valid_moves:
                        tile.unset_valid_frame()
                    #Move piece to dst piece(destination)
                    self._move_piece(dst_piece)
                    #delete
                    # if self._move_piece(dst_piece)=="Break":
                    #     self.selected.unselect()
                    #     self.selected = None
           
            
       
       
    def _move_piece(self,dst_piece):
        """
        Moves selected piece from it's position to dst_pieces and ends
        turn 
        """ 
        board = self.board.get_board()
        self._move_from(self.selected)
        self._move_to(dst_piece)
        self._change_turns()
        self._is_game_over()
        self.turn_number+=1

    
        # piece was out of playground put it ingame
        if self.selected in board["white_pieces_out"]:
            board["white_pieces_in"].append(self.selected)
            board["white_pieces_out"].remove(self.selected)
        elif self.selected in board["black_pieces_out"]:
            board["black_pieces_in"].append(self.selected)
            board["black_pieces_out"].remove(self.selected)

        #unselect piece at the and
        self.selected.unselect()
        self.selected = None

    def _move_from(self, selected):
        """
        Takes care of piece selected to be moved.
        Adds free tile in it's position, deletes sufficent free tiles
        """
        
        if selected.in_play:
            #Beetle case
            if selected.type == "Beetle":
                if selected.piece_under: 
                    #if moved Beetle was on top of another piece
                    selected.piece_under.piece_on_top = None
                    selected.piece_under = None
                    return
                
            #In other case, put Free tile on moved piece place
            freed_tile = Free(selected.get_coordinates(),"x",["x","x","x","x","x","x"])
           
            for index,neighbour in enumerate(selected.neighbours):          
                neighbour.n_cnt-=1
                if neighbour.n_cnt == 0 and neighbour.type == "Free": 
                    #Free neighbour piece doesn't have any other real neihgbours, delete it
                    self.board.get_board()["free_tiles"].remove(neighbour)

                    #when we delete neighbour we need to take it from  neighbours list of adjecent tiles
                    for i,n in enumerate(neighbour.neighbours):
                        if n != "x":
                            n.neighbours[(i+3)%6] = "x"


                else:
                    #If neighbour is not Free tile with zero real neighbours, connect it with free_tile
                    neighbour.neighbours[(index+3)%6] = freed_tile
                    freed_tile.neighbours[index] = neighbour
                    if neighbour.type!="Free":
                        freed_tile.n_cnt+=1
            
            #Add free tile to position of selected tile
            freed_tile.set_position(selected.get_position())
            self.board.get_board()["free_tiles"].append(freed_tile)

        else:
            #If selected piece is not in game, put it in game
            selected.in_play = True
            if selected.player=="w":
                self.board.get_board()["white_pieces_in"].append(self.selected)
                self.board.get_board()["white_pieces_out"].remove(self.selected)
            else:
                self.board.get_board()["black_pieces_in"].append(self.selected)
                self.board.get_board()["black_pieces_out"].remove(self.selected)

    def _move_to(self, dst_piece):
        """
        Moves selected piece to dst_piece and takes care of pieces mutal connections
        """

        #overtake neighbours of dst_piece,position, coordinats and give it to selected piece
        self.selected.set_neighbours(dst_piece.neighbours) 
        self.selected.set_position(dst_piece.get_position())
        self.selected.set_coordinates(dst_piece.coordinates)
        #Beetle case
        beetle_case = False
        if  dst_piece in self.board.board["free_tiles"]:
            self.board.get_board()["free_tiles"].remove(dst_piece)
        else:
            #if dst_piece is not in free tiles, it means we are moving beetle on top of hive
            highest_piece = dst_piece.find_highest_piece()
            beetle_case = True
            self.selected.piece_under = highest_piece
            dst_piece.piece_on_top = self.selected
          
        #Take care of free tiles around piece
        if not beetle_case: #if we are moving beetle on top of hive not needed
            for index,neighbour in enumerate(self.selected.neighbours): # add free tiles where needed and reconnect neighbours to selected piece
                if neighbour == "x":
                    self.board.add_free_tile(Piece.sum_coordinates(self.selected,hex_norm[index]),self.selected)
                else:

                    neighbour.neighbours[(index+3)%6] = self.selected #maybe sufficient
                    neighbour.n_cnt+=1  #not needed because we do this in correct_neighbours

            for index,neighbour in enumerate(self.selected.neighbours): #connect neighbours with one another, make sure everything is right
                if neighbour.type =="Free":
                    self.board.correct_neighbours(neighbour)

    def _is_game_over(self):
        if self.white_queen.n_cnt ==6:
            print("Black won!")
            self.game_over = True
        elif self.black_queen.n_cnt==6:
            print("White won!")
            self.game_over = True

    def _change_turns(self):
        """
        Changes turns between white and black
        """
        if self.turn == "w":
            self.turn = "b"
        else:
            self.turn = "w"


    
            
            
            
        


