import cv2
import numpy as np
from seams import *
from reduceWidth import reduceWidth

def reduceHeight(img, numPixels, energy_type='magnitude', k=3):
    
    # Convert the input image to a numpy array
    img = np.array(img)
    img = np.rot90(img, k=1)

    img = reduceWidth(img, numPixels, energy_type=energy_type, k=k)
    img = np.rot90(img, k=3)
    return img 


