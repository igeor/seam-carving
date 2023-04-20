import numpy as np
import cv2

def compute_magnitude(img, k=3):

    # Compute the x and y gradients using Sobel operators on each color channel
    grad_x_r = cv2.Sobel(img[:,:,0], cv2.CV_64F, 1, 0, ksize=k)
    grad_y_r = cv2.Sobel(img[:,:,0], cv2.CV_64F, 0, 1, ksize=k)
    grad_x_g = cv2.Sobel(img[:,:,1], cv2.CV_64F, 1, 0, ksize=k)
    grad_y_g = cv2.Sobel(img[:,:,1], cv2.CV_64F, 0, 1, ksize=k)
    grad_x_b = cv2.Sobel(img[:,:,2], cv2.CV_64F, 1, 0, ksize=k)
    grad_y_b = cv2.Sobel(img[:,:,2], cv2.CV_64F, 0, 1, ksize=k)

    # Compute the energy function at each pixel using the magnitude of the x and y gradients for each color channel
    energy_r = np.abs(grad_x_r) + np.abs(grad_y_r)
    energy_g = np.abs(grad_x_g) + np.abs(grad_y_g)
    energy_b = np.abs(grad_x_b) + np.abs(grad_y_b)

    # Compute the total energy by summing the energy across the three color channels
    energy = energy_r + energy_g + energy_b

    return energy
    

def compute_laplacian(img, k=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a Laplacian filter kernel to the grayscale image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)

    # Compute the absolute value of the Laplacian to get the energy map
    energy_map = np.abs(laplacian)

    return energy_map
