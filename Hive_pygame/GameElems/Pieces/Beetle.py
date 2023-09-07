from  GameElems.Pieces.Piece import Piece

class Beetle(Piece):
    """
    Represents Beetle figure
    """
    def __init__(self,coordinates, player, neighbours, piece_id ):
        super().__init__("Beetle", coordinates, player, neighbours)
        self.in_play = False #if piece is in play ==True
        self.id = piece_id   #needed for faster graph alghoritms (0 to 21)
        self.piece_under = None
    def valid_moves(self,board,turn,turn_number):
        #case when adding piece that is out of game
        list1 = super().valid_moves(board,turn,turn_number)
        if list1:
            return list1   
        available_tiles = []
        real_neighbours = []#Bag fix
        #searching for valid tiles
        for index,neighbour in enumerate(self.neighbours):
            if neighbour.type!="Free":#Bag fix
                real_neighbours.append(neighbour)#Bag fix
            if neighbour.type =="Free":
                if not self.piece_under:
                    if self.is_needle_ears(index):
                        continue
            piece_to_add = neighbour
            if neighbour.piece_on_top:
                piece_to_add = neighbour.find_highest_piece()
            available_tiles.append(piece_to_add)
        

        #Remove tiles that break hive
        valid_tiles = []

       

        for tile in available_tiles:
            if self.piece_under:#if beetle is on top of piece it will not break hive
                valid_tiles.append(tile)
                tile.set_valid_frame()

            elif tile.type!="Free":#Bag fix
                valid_tiles.append(tile)#Bag fix
                tile.set_valid_frame()#Bag fix
            else:
                for real_neighbour in real_neighbours:#Bag fix
                    if tile in real_neighbour.neighbours:#Bag fix
                        valid_tiles.append(tile)
                        #TODO: visualize
                        tile.set_valid_frame()
                        break
        available_tiles = valid_tiles

        return available_tiles