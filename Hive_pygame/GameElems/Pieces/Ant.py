from  GameElems.Pieces.Piece import Piece

class Ant(Piece):
    """
    Represents Ant figure
    """
    def __init__(self,coordinates, player, neighbours, piece_id ):
        super().__init__("Ant", coordinates, player, neighbours)
        self.in_play = False #if piece is in play 
        self.id = piece_id   #needed for faster graph alghoritms (0 to 21)

    def valid_moves(self,board,turn,turn_number):


        list = super().valid_moves(board,turn,turn_number)
        if list:
            return list

        avaliable_tiles = []
        start = self

        if start: 
            queue = []
            queue.append(start)

            while queue:
                s = queue.pop(0)
                for index, neighbour in enumerate(s.neighbours):
                    if neighbour!="x":
                        if neighbour.type =="Free":
                            #check if passable(needle ears)
                            if s.is_needle_ears(index):
                                continue
                            else:
                                if not neighbour.visited:
                                    queue.append(neighbour)
                                    neighbour.visited = True
                                    avaliable_tiles.append(neighbour)
            #unvisit Free Tiles and remove tiles that break hive
            valid_tiles = []
            for tile in avaliable_tiles:
                tile.visited = False
                if tile.n_cnt==1 and (self in tile.neighbours):
                    continue
                else:
                    valid_tiles.append(tile)
                
            avaliable_tiles = valid_tiles
            #TODO: visualize
            for tile in avaliable_tiles:
                tile.set_valid_frame()

            return avaliable_tiles
        
        else:
            return []
        
        


