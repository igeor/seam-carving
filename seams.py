import numpy as np
from energy_functions import compute_magnitude
from energy_map import *

def find_vertical_seam(img, energy_type='magnitude', k=3):
    """
    Find the optimal vertical seam in an image.
    
    params img: The image to find the seam in.
    returns: A list of integers of the optimal vertical seam.
    
    The output of a vertical seam is a list of integers that represent 
    the indices of the pixels in each row of the image that belong to the optimal vertical seam. 
    """
    h, w, c = img.shape
    
    M = compute_vertical_cumulative_energy_map(img, energy_type=energy_type, k=k)
    
    # Find the optimal vertical seam
    seam = np.zeros(h, dtype=np.int64)
    seam[-1] = np.argmin(M[-1,:])
    for i in range(h-2, -1, -1):
        j = seam[i+1]
        if j == 0:
            seam[i] = np.argmin(M[i,j:j+2]) + j
        elif j == w-1:
            seam[i] = np.argmin(M[i,j-1:j+1]) + j - 1
        else:
            seam[i] = np.argmin(M[i,j-1:j+2]) + j - 1

    return seam

def remove_vertical_seam(img, seam):
    """
    Remove a vertical seam from an image.
    
    params img: The input image as a numpy array.
    params seam: A list of integers representing the indices of the pixels in each row of the image that belong to the seam to remove.
    returns: The output image as a numpy array.
    """
    # Calculate the dimensions of the input image
    h, w, c = img.shape
    
    # Create a new image with one less column
    img_out = np.zeros((h, w - 1, c), dtype=img.dtype)
    
    # Remove the seam from the image
    for i in range(h):
        j = seam[i]
        img_out[i,:,:] = np.delete(img[i,:,:], j, axis=0)
        
    return img_out


def find_horizontal_seam(img, energy_type='magnitude', k=3):
    """
    Find the optimal horizontal seam in an image.
    
    params img: The image to find the seam in.
    returns: A list of integers of the optimal horizontal seam.
    
    The output of a horizontal seam is a list of integers that represent 
    the indices of the pixels in each column of the image that belong to the optimal horizontal seam. 
    """
    img = np.rot90(img, k=1)
    seam = find_vertical_seam(img, energy_type=energy_type, k=k)

    # Reverse the seam list and return it
    return seam[::-1]

   

def remove_horizontal_seam(img, seam):
    """ 
    Remove a horizontal seam from an image.
    
    params img: The input image as a numpy array.
    params seam: A list of integers representing the indices of the pixels in each column of the image that belong to the seam to remove.
    returns: The output image as a numpy array.
    """

    # Get the height and width of the image
    h, w, c = img.shape
    
    # Create a new image with one less row
    img_out = np.zeros((h - 1, w, c), dtype=img.dtype)
    
    for j in range(w):
        row_in = 0
        row_out = 0
        while row_in < h and row_out < h-1:
            if row_in == seam[j]:
                row_in += 1
            img_out[row_out, j] = img[row_in, j]
            row_in += 1
            row_out += 1
    
    return img_out
    
    