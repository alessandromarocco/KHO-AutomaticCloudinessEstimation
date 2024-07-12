"""
A script to define functions used to give an estimation of the cloudiness according to the Red and Blue channel of
the image

Author: Alessandro Marocco
Email: alessandro.marocco@ens.psl.eu
Date: July 2024
"""

# Import the required modules
import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from image_tools import cmask, masked_image, combine_masks, CCR
from otsu_estimation import estimation_Otsu

def R_correction(img):
    """
    Apply a correction to the Red channel of an RGB image by substracting 25 (and put 0 if the result is negative)
    :param img: array, RGB image
    :return: array, RGB image with R correction
    """
    [rows, cols] = img.shape[0], img.shape[1]
    img_r = img.copy()

    for i in range(0, rows):
        for j in range(0, cols):
            if img[i, j, 2] < 25:
                img_r[i, j, 2] = 0
            else:
                img_r[i, j, 2] = img[i, j, 2] - 25

    return img_r

def normalized(image):
    """
    Normalize an image by setting its values between 0 and 255
    :param image: array, image to normalize
    :return: array, image normalized between 0 and 255
    """

    [rows, cols] = image.shape[0], image.shape[1]
    maxi, mini = np.max(image), np.min(image)
    imageNorm = np.zeros((rows,cols))
    for i in range(0, rows):
        for j in range(0, cols):
            imageNorm[i][j] = int(((image[i][j] - mini) / (maxi - mini)) * 255)

    imageNorm = imageNorm.astype("uint8")

    return imageNorm

def bimodality_coefficient(image):
    """
    Compute the bimodality coefficient of a one channel image (Pfister et al., 2013)
    :param image: array, one channel image to calculate the bimodality coefficient
    :return: float, bimodality coefficient result
    """
    data = np.array(image).flatten()
    n = len(data)
    m3 = skew(data,bias=False)
    m4 = kurtosis(data, fisher=True, bias=False)

    bc = (m3**2 + 1) / (m4 + 3 * ((n - 1)**2) / ((n - 2) * (n - 3)))

    return bc

def create_RBmap(image,region):
    """
    Create and return the Red Blue map with the highest Bimodality Coefficient between the Red Blue Ratio, Red Blue
    Difference and Red Blue ratio Normalized (cf Qingyong L. and Weitao L., 2011 for the normalized ratio)
    :param image: array, preprocessed image of size 480 x 480 pixels to apply the cloud cover estimation on
    :param region: array, Mask image with the region where to segment the image has 1 value
    :return: array, The Red Blue map (480x480 pixels) with the highest Bimodality Coefficient
    """

    [rows, cols] = image.shape[0], image.shape[1]

    # Compute the Red Blue Ratio, Red Blue Difference and Red Blue ratio Normalized
    RBRmap = np.zeros((rows, cols))
    RBDmap = np.zeros((rows, cols))
    RBnormalized = np.zeros((rows, cols))

    for i in range(0, rows):
        for j in range(0, cols):
            if region[i][j] == 1:
                if image[i][j][0] != 0:

                    RBRmap[i][j] = int(image[i][j][2]) / int(image[i][j][0])

                    RBDmap[i][j] = np.abs(int(image[i][j][2]) - int(image[i][j][0]))

                    RBnormalized[i][j] = (int(image[i][j][2]) - int(image[i][j][0])) / (
                                int(image[i][j][2]) + int(image[i][j][0]))

    labels = ["Red Blue Ratio", "Red Blue Difference", "Red Blue Ratio Normalized"]

    # Range the result maps between 0 and 255
    maps = []
    RBRmap = normalized(RBRmap)
    RBDmap = normalized(RBDmap)
    RBnormalized = normalized(RBnormalized)
    RBnormalized = masked_image(RBnormalized, region)
    maps.append(RBRmap); maps.append(RBDmap); maps.append(RBnormalized)

    # Compute the Bimodality Coefficient for each map
    BC = []
    BC_RBRmap = bimodality_coefficient(RBRmap)
    BC_RBDmap = bimodality_coefficient(RBDmap)
    BC_RBnormalized = bimodality_coefficient(RBnormalized)
    BC.append(BC_RBRmap); BC.append(BC_RBDmap); BC.append(BC_RBnormalized)

    print("The best R/B map is the", labels[BC.index(max(BC))], "with a BC = ", max(BC))

    # The map with the highest Bimodality Coefficient is returned
    return maps[BC.index(max(BC))]

def estimation_RBR(image, moon_mask=None):
    """
    Main function to use in order to estimate the cloud cover of an image by using the Red Blue detection method
    :param image: array, preprocessed image of size 480 x 480 pixels to apply the cloud cover estimation on
    :param moon_mask: array, mask of the Moon on the image with 255 value for the Moon affected area and 0 elsewhere
    :return: (float, array), percentage of the image covered by clouds and the segmented image between clouds (255 value)
    and sky
    """

    # R channel correction
    image_RBR = R_correction(image)

    # Compute the best Red Blue map (i.e with the highest bimodality coefficient)
    RBmap = create_RBmap(image_RBR,cmask([240, 240], 210, image))

    # Otsu thresholding over the best Red Blue map
    imageProc = estimation_Otsu(RBmap)[1]

    # Applying the masks on the image
    if type(moon_mask) != type(None):
        imageProc = masked_image(imageProc, moon_mask, fill=0, color=255)
    imageProc = masked_image(imageProc, cmask([240, 240], 210, imageProc))

    # Compute the Cloud Cover Index from the image
    CCI = CCR(imageProc)

    return round(CCI,2), imageProc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.io import imread

    #img_path = "test_images/intermediate/LYR-Sony-290220_181403-ql.jpg"
    img_path = "test_images/intermediate/LYR-Sony-080220_083852-ql.jpg"
    #img_path = "test_images/intermediate/LYR-Sony-080220_003742-ql.jpg"
    img = imread(img_path)[:-24]

    CCI, imgProc = estimation_RBR(img)

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(imgProc,cmap="gray")
    plt.title("Red Blue segmentation")
    plt.show()


