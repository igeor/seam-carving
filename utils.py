import numpy as np 
import cv2 
import matplotlib.pyplot as plt


# Read the input image
def read_image(filename, gray=False):
    # read a black and white image as rgb 
    img = plt.imread(filename)
    if gray:
        # Convert the grayscale image to an RGB image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def show_image(img):
    # show as rgb image
    plt.imshow(img, cmap='viridis')
    # remove the axis
    plt.axis('off')
    
def show_grid(imgs, titles=['img1', 'img2', 'img3'], axis='off'):
    fig, axes = plt.subplots(1, len(imgs), figsize=(20, 20))
    for i, img in enumerate(imgs):
        # add border to axes[i]
        axes[i].imshow(img, cmap='viridis')
        if not axis:
            axes[i].axis('off')
        axes[i].set_title(titles[i])
        
def add_vseam_to_img(img, seam, color=(0, 0, 255)):
    img = np.copy(img)
    h, w, c = img.shape
    for i in range(h):
        j = seam[i]
        img[i, j, :] = color
    return img 

def add_hseam_to_img(img, seam, color=(0, 0, 255)):
    img = np.copy(img)
    h, w, c = img.shape
    for i in range(w):
        j = seam[i]
        img[j, i, :] = color    
    return img