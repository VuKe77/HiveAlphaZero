from  GameElems.Pieces.Piece import Piece
from GameElems.game_utils import HiveUtils
class Free(Piece):
    """
    Represents Free piece which is available in game for putting piece on it.
    """
    #I have decided to implement this class becouse it makes sense to have pieces that are available, because board is dynamic
    def __init__(self,coordinates, player, neighbours ):
        super().__init__("Free", coordinates, player, neighbours)
        self.visited = False
