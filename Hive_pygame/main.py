import pygame as pg
from GameElems.constants import BACKGROUND,SCREENHEIGHT,SCREENWIDTH, FPS
from GameElems.Hive_Game import Game



def run():
    #Initialise screen
    pg.init()
    WIN = pg.display.set_mode((SCREENWIDTH,SCREENHEIGHT))
    pg.display.set_caption("Hive")
   
    #initialise clock
    clock = pg.time.Clock()
    game = Game(WIN)
    #Event loop
    running = True
    while running:
        clock.tick(60)  
        
        #event polishing handler
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False
            elif event.type == pg.MOUSEBUTTONUP:
                game.select_piece(event.pos)
            elif event.type == pg.KEYUP:
                if event.key ==pg.K_r   :
                    game.restart_game()

        game.update()

    pg.quit()

run()