import pygame as pg
import os


class HiveUtils:

   

    @staticmethod
    def load_png(name):
        """ Load image and return image object"""
        main_dir = os.path.split(os.path.abspath(__file__))[0]
        img_dir = os.path.join(main_dir, "images")
        image_path = os.path.join( img_dir, name+".png")

        image = pg.image.load(image_path)
        if image.get_alpha() is None:
            image = image.convert()
        else:
            image = image.convert_alpha()
        return image, image.get_rect()
