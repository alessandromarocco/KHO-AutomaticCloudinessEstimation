"""
A script to define functions used to give an estimation of the cloudiness according using the Otsu's thresholding
method for a single channel image

Author: Alessandro Marocco
Email: alessandro.marocco@ens.psl.eu
Date: July 2024
"""

# Import the required modules
import numpy as np
import cv2
import squircle

from image_tools import CCR
from stars_estimation import is_star

def remove_stars(image):
    """
    Remove the stars from the image
    :param image: array, image from which removing the stars
    :return: array, image with the detected stars removed
    """
    [rows, cols] = image.shape
    img_stars = image.copy()
    thresh = 20
    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            if img_stars[i, j] > thresh and is_star(image, [i, j]) == True:
                image[i, j] = 0

    return image

def back_to_original_size(image):
    """
    Put an image of 416 x 416 pixels in the center of an image of 480 x 480 pixels
    :param image: array, image to put in the orignal size
    :return: array, image with the orignal size of 480 x 480 pixzlq
    """

    original = np.zeros((480, 480))
    for i in range(32, 448):
        for j in range(32, 448):
            original[i][j] = image[i - 32][j - 32]

    return original


def estimation_Otsu(image):
    """
    Main function to use in order to estimate the cloud cover of an image by using the Otsu thresholding method

    :param image: array, preprocessed image of size 480 x 480 pixels to apply the cloud cover estimation on
    :return: (float, array), percentage of the image covered by clouds and the segmented image between clouds (255 value)
    and sky
    """
    # Convert the RGB image to Grayscale image if it's not already one
    if len(np.shape(image)) != 2:
        image_otsu = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_otsu = image

    # Remove the stars from the image (bright points)
    image_otsu = remove_stars(image_otsu)

    # Transform the image into a square, to remove the black part outside the circle
    image_otsu = squircle.to_square(image_otsu, method="stretch")[32:448, 32:448]  # Circle transformed to a square

    # Apply a Gaussian Blur to have a smoothest image
    image_otsu = cv2.GaussianBlur(image_otsu, (3, 3), 0)

    # Apply the Otsu thresholding mehtod
    thr, image_otsu = cv2.threshold(image_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Transform back the image into a circle
    image_otsu = squircle.to_circle(image_otsu, method="stretch")  # Square transformed back to a circle

    # Back to original size of the image
    image_otsu = back_to_original_size(image_otsu)

    # Compute the Cloud Cover Index from the image
    CCI = CCR(image_otsu)

    print("Otsu threshold = ", thr)
    return round(CCI,2), image_otsu

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.io import imread

    img_path = "test_images/intermediate/LYR-Sony-261219_181225-ql.jpg"

    img = imread(img_path)[:-24]

    CCI, imgProc = estimation_Otsu(img)

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(imgProc,cmap="gray")
    plt.title("Otsu's tresholding")
    plt.show()

