import cv2
import numpy as np

def display_seam(img, seam, seam_direction='vertical'):
    """
    Display the selected seam on top of the original image.

    :param img: The original image.
    :param seam: The seam to be displayed.
    :param seam_direction: The direction of the seam (either 'vertical' or 'horizontal').
    """
    img = cv2.imread(img_filename)

    if seam_direction == 'vertical':
        # Draw the selected seam on top of the original image
        for row in range(img.shape[0]):
            col = seam[row]
            img[row, col] = [0, 0, 255]

    elif seam_direction == 'horizontal':
        # Transpose the image and the seam for horizontal seams
        img = img.transpose((1, 0, 2))
        seam = seam[::-1]

        # Draw the selected seam on top of the transposed image
        for row in range(img.shape[0]):
            col = seam[row]
            img[row, col] = [0, 0, 255]

        # Transpose the image back to its original orientation
        img = img.transpose((1, 0, 2))

    # Display the original image and the selected seam
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
