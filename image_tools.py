"""
A script to define image tools functions (masking, resize, crop) used for cloud cover estimation from Kjell Henriksen
Observatory nighttime images.

Author: Alessandro Marocco
Email: alessandro.marocco@ens.psl.eu
Date: July 2024
"""

# Import the required modules
import numpy as np
import cv2


def cmask(index, radius, array):
    """
    Build a mask for an image from the center position of the circle and its radius wanted
    :param index: tuple, position of the center of the circle to draw on the mask image
    :param radius: int, radius of the circle to draw
    :param array: array, image reference to build the mask image with the same size
    :return: array, mask image with 1 value for the circle and 0 value for the rest of the image
    """
    a, b = index
    is_rgb = len(array.shape)

    if is_rgb == 3:  # if image has 3 channels
        ash = array.shape
        nx = ash[0]
        ny = ash[1]

    else:  # if image has only one channel
        nx, ny = array.shape

    s = (nx, ny)
    image_mask = np.zeros(s)
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= radius * radius
    image_mask[mask] = 1

    return image_mask


def masked_image(image, mask, fill=1, color=0):
    """
    Mask an image using a mask, can choose if the 0 or 1 values of the mask are masked and choose the color of the mask
    on the image
    :param image: array, image to mask
    :param mask: array, mask used to mask the image
    :param fill: int, 0 or 1 to choose if we want to mask the image where it's 0 or where it's 1 on the mask
    :param color: int, 0 or 255 to choose the color of the masked part on the image (0 black and 255 for white)
    :return: array, masked image
    """

    [rows, cols] = mask.shape
    image_masked = np.ones([rows, cols]) * color

    if len(image.shape) == 3:
        channels = image.shape[2]
        image_masked = np.ones([rows, cols, channels]) * color

    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i, j] == fill:
                if len(image.shape) == 3:
                    for k in range(channels):
                        image_masked[i, j][k] = image[i, j][k]
                else:
                    image_masked[i][j] = image[i][j]

    return image_masked.astype("uint8")


def combine_masks(moon_mask, circle_mask):
    """
    Combine the moon mask and the circle mask of interest in order to keep only the area of interest without the moon
    affected area
    :param moon_mask: array, Mask corresponding to the area affected by the moon
    :param circle_mask: array, Mask corresponding to the area of interest on the image
    :return: combined_mask: array, Mask corresponding to the area of interest excluding the area affected by the moon
    """
    [rows, cols] = circle_mask.shape

    combined_mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if circle_mask[i][j] == 1 and moon_mask[i][j] == 0:
                combined_mask[i][j] = 1

    return combined_mask

def compute_brightness(img):
    """
    Compute the mean brightness of an RGB image using the channel V of the HSV (hue, saturation, value) channel module
    after conversion
    :param img: array, image to compute the brightness
    :return: float, Mean brightness of the RGB image
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    [rows, cols, channels] = img.shape
    mask = cmask([240, 240], 210, img)
    brightness = []
    for i in range(rows):
        for j in range(cols):
            if mask[i][j] == 1:
                brightness.append(img[i][j][2])

    mean_brightness = np.mean(brightness)

    print("Image brightness: ", mean_brightness)

    return mean_brightness


def preprocessing_image(image):
    """
    Preprocessing the input RGB image in order to be used in the algorithms for cloud coverage estimation
    :param image: array, image (any size but usually 480x480 or 480x504 for KHO)
    :return: image: array, image of size 480x480 masked to keep the area of interest
    """
    img_mask = cmask([240, 240], 210, image)  # Mask with radius 210 pixels at the center of the image [240,240]

    if np.shape(image) == (480, 480, 3):
        image = masked_image(image, img_mask)

    else:  # Case where there is the band at the bottom of the image
        image = masked_image(image, img_mask)[:-24]  # Original image is masked

    return image


def CCR(imageProc):
    """
    Compute the Cloud Coverage Ratio in percentage of an image where clouds have the maximum value of 255.
    :param imageProc: Image with cloudiness
    :return: CCR: float, Cloud Coverage Ratio in percentage
    """
    mask = cmask([240, 240], 210, imageProc)
    [rows, cols] = mask.shape
    tot, countCloud = 0, 0
    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i, j] == 1:
                if imageProc[i, j] == 255:
                    countCloud += 1
                    tot += 1
                else:
                    tot += 1

    return (countCloud / tot) * 100


if __name__ == "__main__":
    from skimage.io import imread
    import matplotlib.pyplot as plt

    img_path = "test_images/intermediate/LYR-Sony-290220_181403-ql.jpg"
    img = imread(img_path)[:-24]
    img = preprocessing_image(img)

    from skimage.transform import resize

    plt.imshow(img)
    plt.show()
    img_160 = resize(img, (160, 160))
    img_96 = resize(img, (96, 96))
    img_48 = resize(img,(48,48))
    plt.imshow(img_160)
    plt.show()
    plt.imshow(img_96)
    plt.show()
    plt.imshow(img_48)
    plt.show()


    # Original image from KHO and axample of a preprocessed image
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")

    img_pre = preprocessing_image(img)
    plt.subplot(122)
    plt.imshow(img_pre)
    plt.title("Preprocessed image")
    plt.show()

    # Example of masks
    mask = cmask([240, 240], 210, img_pre)
    plt.imshow(mask)
    plt.title("Mask example (area of interest)")
    plt.show()

    moon_mask = cmask([150, 150], 80, img_pre)
    plt.imshow(moon_mask)
    plt.title("Mask example (area affected by the Moon)")
    plt.show()

    # Combine masks example
    comb_mask = combine_masks(moon_mask, mask)
    plt.imshow(comb_mask)
    plt.title("Combined mask \n (1: pixels of interest, 0: not taking into account)")
    plt.show()

    # Example of cloud cover ratio on the mask
    print("Cloud Cover of the combined mask: ", CCR(comb_mask), "%")
