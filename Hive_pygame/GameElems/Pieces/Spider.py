from  GameElems.Pieces.Piece import Piece

class Spider(Piece):
    """
    Represents Spider figure
    """
    def __init__(self,coordinates, player, neighbours, piece_id ):
        super().__init__("Spider", coordinates, player, neighbours)
        self.in_play = False #if piece is in play 
        self.id = piece_id   #needed for faster graph alghoritms (0 to 21)

    def valid_moves(self,board,turn,turn_number):
        #case when adding piece that is out of game
        list1 = super().valid_moves(board,turn,turn_number)
        if list1:
            return list1

        valid_paths = set()
        paths=[]
        visited_tiles = set() #in order to unvisit visited tiles at the end
        queue = []
        queue.append([self,-1])
        cnt = -1 




        ###DFS alghoritm that saves paths according to spider movement rules
        while queue:
            s,cnt = queue.pop()
            paths.append(s)
            cnt+=1
            if cnt==3:
                valid_paths.add(paths[-1])
                cnt=0
                continue
            for index,neighbour in enumerate(s.neighbours):
                if neighbour!="x":
                        if neighbour.type =="Free":
                            if s.has_common_neighbour(neighbour,self):           
                                if s.is_needle_ears(index,self):
                                    continue
                                else:
                                    if not neighbour.visited:
                                        queue.append([neighbour,cnt])
                                        visited_tiles.add(neighbour)
                                        neighbour.visited = True
                                        
        #unvisit Free Tiles
        for tiles in visited_tiles:
            tiles.visited = False
        #TODO: visualize
        available_tiles = set()
        for tile in valid_paths:
            tile.set_valid_frame()
            available_tiles.add(tile)
        
        return list(available_tiles)
