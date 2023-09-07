from GameElems.Pieces.Queen import Queen
from GameElems.Pieces.Grasshopper import Grasshopper
from GameElems.Pieces.Beetle import Beetle
from GameElems.Pieces.Ant import Ant
from GameElems.Pieces.Piece import Piece
from GameElems.Pieces.Spider import Spider
from GameElems.constants import SCREENHEIGHT, SCREENWIDTH, BACKGROUND
import pygame as pg
from GameElems.Pieces.Free import Free


"""
Coordinates used in game are cube coordinates. These are directions for every piece
relative to [0,0,0]. Implementation have been also done just with 2 coordinates, because 
x+y+z=0 must be satisfied
"""
n = [1,-1,0]
nw = [0,-1,1]
sw = [-1,0,1]
s = [-1,1,0]
se = [0,1,-1]
ne = [1,0,-1]
hex_norm = [n,nw,sw,s,se,ne]


class Board:
    
    def __init__(self):
        self.board ={"white_pieces_in":[],
                     "black_pieces_in":[],
                     "white_pieces_out":[],
                     "black_pieces_out":[],
                     "free_tiles": []}
        self._create_board()

    def get_board(self):
        return self.board   

    def _create_board(self):
        """Initialize board with figures"""

        #white
        self._add_piece_to_board(0,Queen,"w")
        self._add_piece_to_board(1,Ant,"w")
        self._add_piece_to_board(2,Ant,"w")
        self._add_piece_to_board(3,Ant,"w")
        self._add_piece_to_board(4,Spider,"w")
        self._add_piece_to_board(5,Spider,"w")
        self._add_piece_to_board(6,Grasshopper,"w")
        self._add_piece_to_board(7,Grasshopper,"w")
        self._add_piece_to_board(8,Grasshopper,"w")
        self._add_piece_to_board(9,Beetle,"w")
        self._add_piece_to_board(10,Beetle,"w")
        #black
        self._add_piece_to_board(11,Queen,"b")
        self._add_piece_to_board(12,Ant,"b")
        self._add_piece_to_board(13,Ant,"b")
        self._add_piece_to_board(14,Ant,"b")
        self._add_piece_to_board(15,Spider,"b")
        self._add_piece_to_board(16,Spider,"b")
        self._add_piece_to_board(17,Grasshopper,"b")
        self._add_piece_to_board(18,Grasshopper,"b")
        self._add_piece_to_board(19,Grasshopper,"b")
        self._add_piece_to_board(20,Beetle,"b")
        self._add_piece_to_board(21,Beetle,"b")
        


        self.board["free_tiles"].append(Free([0,0,0],"x",["x","x","x","x","x","x"]))
        screen_position = [SCREENWIDTH/2,SCREENHEIGHT/2]
        self.board["free_tiles"][0].set_position(screen_position)

    def _add_piece_to_board(self,id,type_constructor,player):
        """
        Makes new piece provided with type_constructor, sets up piece id and player 
        provided by function arguments
        """
        total = 11
        spacing = 5
        new_piece = type_constructor([-1,-1,-1],player,["x","x","x","x","x","x"],id)
        width,height =  new_piece.get_sprite_dimensions()
        if player == "w":
            self.board["white_pieces_out"].append(new_piece)
            screen_position = [SCREENWIDTH - total*(width+spacing)+width/2 + id*(width+spacing), spacing+height/2]
        elif player ==  "b":
             self.board["black_pieces_out"].append(new_piece)
             screen_position = [SCREENWIDTH - total*(width+spacing)+width/2 + (id-11)*(width+spacing), 2*spacing+height*3/2]
        else:
            raise TypeError
        
        new_piece.set_position(screen_position)

    def draw(self,win):
        """
        Draws board on screen. Needed for visualization
        """
        win.fill(BACKGROUND)
        font1 = pg.font.Font(None,64)
        text = font1.render("Let's play Hive!",True, (10,10,10))
        textpos = text.get_rect(centerx = SCREENWIDTH/3,y=10)
        win.blit(text, textpos)


        #draw pieces
        for pieces in self.board.values():
            for piece in pieces:
                #Beetles must be drawed on top of figures
                if not piece.piece_on_top:
                    piece.draw(win)

    def add_free_tile(self,coordinates,dst_piece):
 
        """ 
        Adds Free Tile at position given by coordinates argument, free
        tile will be neighbour of dst_piece. 
        """
       # Get right display position for new tile sprite
        height,width = dst_piece.get_frame_dimensions()
        new_position = None
        dst_position = dst_piece.get_position()
        dst_coordinates = dst_piece.get_coordinates()
        delta = [0,0,0]
        direction =None
        for i in range(3):
            delta[i] = coordinates[i] - dst_coordinates[i]
            
        if delta == hex_norm[0]:
            new_position = [dst_position[0],dst_position[1]-height]
            direction =0
        elif delta == hex_norm[1]:
            new_position = [dst_position[0]-3*width/4,dst_position[1]-height/2]
            direction =1
        elif delta == hex_norm[2]:
            new_position = [dst_position[0]-3*width/4,dst_position[1]+height/2]
            direction =2
        elif delta == hex_norm[3]:
            new_position = [dst_position[0],dst_position[1]+height]
            direction =3
        elif delta == hex_norm[4]:
            new_position = [dst_position[0]+3*width/4,dst_position[1]+height/2]
            direction =4
        elif delta == hex_norm[5]:
             new_position = [dst_position[0]+3*width/4,dst_position[1]-height/2]
             direction =5
        
        new_piece = Free(coordinates,"x",["x","x","x","x","x","x"])
        new_piece.set_position(new_position)
        self.board["free_tiles"].append(new_piece)
        new_piece.neighbours[(direction+3)%6] = dst_piece#needed?
        new_piece.n_cnt+=1#needed?
        dst_piece.neighbours[direction] = new_piece #add new_piece as neighbour to dst_piece

    def correct_neighbours(self,target):
        """
        Correct neighbours of target. looks for every piece in Hive(Free tiles included)
        and adds it to target neighbours if needed. 
        """
        neighbour_coords = Piece.coordinates_of_neighbours(target)
        for key, pieces in self.board.items():
            if (key == "white_pieces_out" or key=="black_pieces_out"):
                continue
            for piece in pieces:
                try:
                    index = neighbour_coords.index(piece.coordinates)
                    if target.neighbours[index] == "x":
                        target.neighbours[index] = piece
                        if piece.type !="Free":
                             target.n_cnt+=1
                    
                        piece.neighbours[(index+3)%6] = target
                       
                        if target.type != "Free":
                            piece.n_cnt+=1
                except:
                    pass
    
    def check_hive_connectivity(self, selected_piece):
        """
        Check if hive will be connected when selected_piece is 
        taken out, in order to check for valid moves
        """

       
        #not in hive case
        if (selected_piece in self.board["white_pieces_out"]) or(selected_piece in self.board["black_pieces_out"]):
            return False
        
        #beetle on hive case
        if selected_piece.type == "Beetle":
            if selected_piece.piece_under:
                return False
        
        #on edge of hive case
        if selected_piece.n_cnt ==1:
            return False

        
       

        ###take selected_piece out of the hive
        for index,neighbour in enumerate(selected_piece.neighbours):
            if not (neighbour =="x"):
                if neighbour.type!="Free":
                    neighbour.neighbours[(index+3)%6] = "x"

        ###BFS algorithm for going through tree
        visited = [None]*22
        
  

        queue = []
        #get starting node, must not be selected_piece
        #fill in visited array with False values for pieces in game
        start_valid = False
        start = None
        for key,pieces in self.board.items():
            if key not in ["white_pieces_out", "black_pieces_out","free_tiles"]:
                for piece in pieces:
                    if piece!=selected_piece:
                        if not start_valid:
                            start = piece
                            start_valid = True
                        if piece.type=="Beetle": 
                            #Automaticly visit Beetles on top of hive
                            if piece.piece_under:
                                visited[piece.id] = True 
                            else:
                                visited[piece.id] = False
                        else:
                            visited[piece.id] = False
                        
                
        queue.append(start)
        visited[start.id] = True

        while queue:
                s = queue.pop(0)
                for neighbour in s.neighbours:
                    if neighbour != "x":
                        if neighbour.type != "Free":
                            if visited[neighbour.id] == False:
                                queue.append(neighbour)
                                visited[neighbour.id] = True

        ###return selected_piece into hive
        for index,neighbour in enumerate(selected_piece.neighbours):
            if not (neighbour =="x"):
                if neighbour.type!="Free":
                    neighbour.neighbours[(index+3)%6] = selected_piece


        if False in visited:
            return True # if we haven't visited some piece return True(hive broken)
        else:
            return False
