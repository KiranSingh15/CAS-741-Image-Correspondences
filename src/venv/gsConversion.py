# declarations
import cv2 as cv

def convert_to_greyscale(img_colour):
    img_gs =  cv.cvtColor(img_colour, cv.COLOR_BGR2GRAY)
    return img_gs
