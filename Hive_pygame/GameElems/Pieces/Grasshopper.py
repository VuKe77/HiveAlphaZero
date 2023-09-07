from  GameElems.Pieces.Piece import Piece

class Grasshopper(Piece):
    """
    Represents Grasshopper figure
    """
    def __init__(self,coordinates, player, neighbours, piece_id ):
        super().__init__("Grasshopper", coordinates, player, neighbours)
        self.in_play = False #if piece is in play 
        self.id = piece_id   #needed for faster graph alghoritms (0 to 21)

    def valid_moves(self,board,turn,turn_number):
       #case when adding piece that is out of game
        list1 = super().valid_moves(board,turn,turn_number)
        if list1:
            return list1
        
        available_tiles = []
        #find straight lines of pieces
        for direction,neighbour in enumerate(self.neighbours):
            if neighbour.type!="Free":
                on_direction = [neighbour]
                while on_direction:
                    s = on_direction.pop()
                    next = s.neighbours[direction]
                    if next!="x":
                        if next.type!="Free":
                            on_direction.append(next)
                        else:
                            available_tiles.append(next)
        #TODO: visualize
        for tile in available_tiles:
            tile.set_valid_frame()

        return available_tiles
                


