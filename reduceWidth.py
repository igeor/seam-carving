import cv2
import numpy as np
from seams import *

def reduceWidth(img, numPixels, energy_type='magnitude', k=3):
    
    # Convert the input image to a numpy array
    img = np.array(img)

    # Iterate numPixels times to remove that many seams
    for i in range(numPixels):
        
        # Find the optimal vertical seam
        seam = find_vertical_seam(img, energy_type=energy_type, k=k)
        
        # Remove the seam from the image
        img = remove_vertical_seam(img, seam)

    return img


