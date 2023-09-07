# All game logic is in this file.
Hive_Game represents game of Hive, which incorporates Pieces and Board  
  
  
Pieces are implemented as graph nodes, where each piece has list of it's neighbours, which represents
other pieces.  
Each piece has also it's own coordinates. I have used [cube coordinate system](https://www.redblobgames.com/grids/hexagons/).  
Most of alghoritms are therefore using BFS and DFS implemented logic.

