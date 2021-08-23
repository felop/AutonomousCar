import numpy as np
import cv2
from random import randint

polygnNUMBER = 2
randomizer = 3

def combine_two_color_images_with_anchor(image1, image2, anchor_y, anchor_x):
    foreground, background = image1.copy(), image2.copy()
    # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
    background_height = background.shape[1]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
        raise ValueError("The foreground image exceeds the background boundaries at this location")
    # do composite at specified location
    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y+foreground_height
    end_x = anchor_x+foreground_width

    background[start_y:end_y, start_x:end_x,:] = foreground
    return background

def shadow(img,randomizer=3,polygnNUMBER=2,alpha=0.8):
    # add borders
    size = [img.shape[0],img.shape[1]]
    border = np.zeros((20*2+img.shape[0],20*2+img.shape[1],3)) # create a single channel 200x200 pixel black
    borderedIMG = combine_two_color_images_with_anchor(img,border,20,20)
# add polygones
    for i in range(int(randint(10,10*randomizer)/10*polygnNUMBER)):
        A = [randint(0,20),randint(0,40+img.shape[1])]
        B = [randint(0,40+img.shape[0]),randint(img.shape[1]+20,img.shape[1]+40)]
        C = [randint(img.shape[0]+20,img.shape[0]+40),randint(0,40+img.shape[1])]
        D = [randint(0,40+img.shape[0]),randint(0,20)]
        contours = np.array( [ A, B, C, D ] )  # A B C D
        mask = borderedIMG.copy
        cv2.fillPoly(borderedIMG, pts =[contours], color=(0,0,0))
        borderedIMG = cv2.addWeighted(mask, alpha, borderedIMG, 1 - alpha, 0)
    img = borderedIMG[20:20+img.shape[0], 20:20+img.shape[1],:]
    return img
