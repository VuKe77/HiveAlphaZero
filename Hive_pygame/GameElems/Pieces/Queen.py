from  GameElems.Pieces.Piece import Piece

class Queen(Piece):
    """
    Represents Queen figure
    """
    def __init__(self,coordinates, player, neighbours, piece_id ):
        super().__init__("Queen", coordinates, player, neighbours)
        self.in_play = False #if piece is in play 
        self.id = piece_id   #needed for faster graph alghoritms (0 to 21)

    def valid_moves(self,board,turn,turn_number):
        #case when adding piece that is out of game
        list1 = super().valid_moves(board,turn,turn_number)
        if list1:
            return list1   
        available_tiles = []
        real_neighbours = []#Bag fix
        for index, neighbour in enumerate(self.neighbours):
            if neighbour.type =="Free":
                if self.is_needle_ears(index):
                    continue
                else:
                    available_tiles.append(neighbour)
            else:
                real_neighbours.append(neighbour)#Bag fix
                


        #Remove tiles that break hive
        valid_tiles= []
        for tile in available_tiles:
            for real_neighbour in real_neighbours:#Bag fix
                if tile in real_neighbour.neighbours:#Bag fix
                    #TODO: visualize
                    valid_tiles.append(tile)
                    tile.set_valid_frame()
        available_tiles= valid_tiles

        return available_tiles