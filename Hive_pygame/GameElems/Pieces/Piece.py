import pygame as pg
from GameElems.game_utils import HiveUtils

n = [1,-1,0]
nw = [0,-1,1]
sw = [-1,0,1]
s = [-1,1,0]
se = [0,1,-1]
ne = [1,0,-1]
hex_norm = [n,nw,sw,s,se,ne]
class Piece:
    """
    Parent class for all pieces: Ant, Beetle, Grasshopper, Queen, Spider, Free
    """

    def __init__(self, type, coordinates, player, neighbours):
        self.type = type # "Free", "Queen"
        self.coordinates = coordinates
        self.player  = player #"w"-White, "b"-Black
        self.neighbours = neighbours #piece neighbours:[n,nw,sw,s,se,ne]
        self.n_cnt = 0 #number of real neighbours(real pieces like Queen, Ant...)
        self.piece_on_top = None #Pointing to the piece above itself

        for piece in self.neighbours:
            if piece !="x":
                if piece.type!="Free":
                    self.n_cnt+=1        
        #sprites initialization
        self._image, self._rect = HiveUtils.load_png(type)
        if self.player == "b":
            self.piece_frame_img, self.piece_frame_rect  = HiveUtils.load_png("b_frame")
        elif self.player  =="w":
            self.piece_frame_img, self.piece_frame_rect  = HiveUtils.load_png("w_frame")
        else:
            self.piece_frame_img, self.piece_frame_rect  = HiveUtils.load_png("f_frame")

        self._selected_frame_img, dummy = HiveUtils.load_png("s_frame") 
        self._current_frame_img = self.piece_frame_img
        self._valid_frame, dummy = HiveUtils.load_png("valid_frame")
        
    
    def get_position(self):
        return [self._rect.centerx, self._rect.centery]
    def get_coordinates(self):
        return self.coordinates
    def get_sprite_dimensions(self):
        return (self.piece_frame_rect.width, self.piece_frame_rect.height)
    def get_frame_dimensions(self):
        return (self.piece_frame_rect.height, self.piece_frame_rect.width)
    def get_neighbours(self):
        return self.neighbours
    def set_neighbours(self, neighbours):
        """
        Set neighbours to piece
        """
        self.neighbours = neighbours
        self.n_cnt = 0
        for piece in self.neighbours:
            if piece !="x":
                if piece.type!="Free":
                    self.n_cnt+=1   

        
        
    


    def set_position(self, new_position):
        self._rect.centerx = new_position[0]
        self._rect.centery = new_position[1]
        self.piece_frame_rect.centerx = new_position[0]
        self.piece_frame_rect.centery = new_position[1]

   
    def set_coordinates(self,new_coordinates):
        self.coordinates = new_coordinates

    def normalize_coordinates(self, target_coordinates):
        """
        Normalizes coordinates of piece with respect to target piece,
        so that it's coordinates take values of hex_norm vector
        """
        norm_coord = [[],[],[],[],[],[]]
        for i in range(len(norm_coord)):
            delta = [-1,-1,-1]
            for j in range(3):
                delta[j] = self.coordinates[j] - target_coordinates[j]
            norm_coord[i] = delta

        return norm_coord
                
   


    def select(self):
        """
        Change piece frame and print out selected piece
        """
        #changes frame when piece is selected
        self._current_frame_img = self._selected_frame_img
        if self.type == "Beetle":
            self._print_beetle()
        else:
             print(self.type+"("+self.player+")"+str(self.coordinates))
    def _print_beetle(self):
        """
        Function for printing ut when beetle is selected
        """
        pieces_under = []
        next = self.piece_under
        while next:
            pieces_under.append(next)
            if next.type =="Beetle":    
                next = next.piece_under
            else:
                break
        print(self.type+"("+self.player+")"+str(self.coordinates)+", pieces_under: " + str(pieces_under))

    def unselect(self):
        self._current_frame_img = self.piece_frame_img
        

    def draw(self, win):
        win.blit(self._current_frame_img,self.piece_frame_rect.topleft)
        win.blit(self._image,self._rect.topleft)

    def valid_moves(self, board,turn,turn_number):
        """
        Returns list of valid turns on board for selected piece
        given board, who's turn it is ("w"/"b") and number of turn in game
        """
        available_tiles =[]

        #Exception to game rule written below: First black move
        if turn_number==1:
            available_tiles= board["free_tiles"]

        if self.in_play ==False:
            #Game rule: When adding piece that is out of game, must not touch opponents piece
            for tile in board["free_tiles"]:
                is_valid = True
                for neighbour in tile.neighbours:
                    if neighbour!="x":
                        neighbour = neighbour.find_highest_piece()
                        if (neighbour.player==turn or neighbour.player=="x"):
                            continue
                        else:
                            is_valid = False
                            break
                if is_valid:
                    available_tiles.append(tile)
        #TODO:Visualize
        for tile in available_tiles:
            tile.set_valid_frame()

        return available_tiles



    def is_needle_ears(self,index,masked_piece=None):
        """
        Returns if neighbour on provided index is reachable
        from piece upon which function was called(self)
        Optinally function can be provided with masked_piece if we wan't
        to mask specific piece when looking for needle ears ( e.g valid moves for Spider)
        """
        left = self.neighbours[(index-1)%6]
        right = self.neighbours[(index+1)%6]

        if (left!="x") and (right!="x"):
            if (left.type!="Free") and (right.type!="Free"):
                if masked_piece:
                    if left==masked_piece or right==masked_piece:
                        return False
                return True
            return False
    def has_common_neighbour(self,piece,masked_piece=None):
        """
        Returns True if self and piece have same real piece neighbour,
        else returns false. Maksed piece is excluded as common neighbour
        """
        for neighbour in self.neighbours:
            if neighbour!="x":
                if neighbour.type!="Free" and neighbour!=masked_piece:

                    if neighbour in piece.neighbours:
                        return True


        return False

    def find_highest_piece(self):
        """
        Finds highest piece on dst_piece coordinates
        """
        highest_piece = self
        while highest_piece.piece_on_top:
            highest_piece = highest_piece.piece_on_top
        return highest_piece

    def set_valid_frame(self):
        self._current_frame_img = self._valid_frame
    def unset_valid_frame(self):
        self._current_frame_img = self.piece_frame_img

    @staticmethod
    def sum_coordinates(piece,direction):
        #TODO: umesto piece moze direktno koordinate ako ubrzava program
        piece_coord = piece.get_coordinates()
        result = [0,0,0]
        for i in range(3):
            result[i]= piece_coord[i]+direction[i]
        return result
    @staticmethod
    def coordinates_of_neighbours(piece):
        """
        Returns coordinates of pieces neighbours
        """
        result = [0,0,0,0,0,0]
        for i in range(6):
            result[i] = Piece.sum_coordinates(piece,hex_norm[i])
        return result
    
    def __repr__(self):
        return f"Piece:{self.type}{self.coordinates}{self.player}"

    


    

