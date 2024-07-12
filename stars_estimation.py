"""
A script to define functions used to identifies stars and give an estimation of the cloudiness according to the stars
detected

Author: Alessandro Marocco
Email: alessandro.marocco@ens.psl.eu
Date: July 2024
"""

# Import the required modules
import cv2
import numpy as np

from image_tools import cmask, masked_image, CCR


def is_star(img, pos, k1=0.8, kmean=0.7):
    """
    Compute if a pixel on the image is a star or not according to the relative brightness of its neighbors
    :param img: array, image where the pixel to analyze is
    :param pos: tuple, position x and y of the pixel to analyze on the image
    :param k1: float, factor applied to the brightness to each neighbor
    :param kmean: float, factor applied to the mean brightness of the neighborhood
    :return: bool, True if the pixel is considered as a star and False if not
    """
    # Define the position, the brightness and the neighborhood of the pixel to analyze
    x, y = pos[0], pos[1]

    brightness = img[x, y]

    neighborhood = [img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1],
                    img[x, y - 1], img[x, y + 1],
                    img[x - 1, y - 1], img[x - 1, y],
                    img[x - 1, y + 1]]

    # Browse all the neighbor
    mean, n = 0, len(neighborhood)

    for neighbor in neighborhood:

        mean += neighbor / n

        # Not a star if a neighbor has a brightness above k1 time the brightness of the pixel to analyze
        if neighbor > k1 * brightness:
            return False

    # Not a star if the mean brightness of the neighborhood is above kmean time the brightness of the pixel to analyze
    if mean > kmean * brightness:
        return False

    return True


def create_mesh(grid_size):
    """
    Create a grid of size 420 x 420 pixels, inside a 480 x 480 image, where each square of the grid has a value
    corresponding to its number from top to right
    :param grid_size: int, number of squares on a grid line (must be an integer divisor of 420)
    :return: array, 480 x 480 pixels image with a 420 x 420 pixels grid where each square of the grid has a different value
    """
    # Increment, which is the length of one side of a square
    incr = int(420 / grid_size)

    # Define the coordinates of the upper left corner of each square of the grid
    mesh = []
    for i in range(0, grid_size + 1):
        line = []
        for j in range(0, grid_size + 1):
            line.append([30 + j * incr, 30 + i * incr])
        mesh.append(line)

    # Assigns to each square in the grid a value corresponding to its number from top to right using the coordinates of
    # the upper left corner of each square of the grid
    squares = np.zeros((480, 480))
    nSquare = 0
    for i in range(grid_size):
        for j in range(grid_size):
            cv2.rectangle(squares, mesh[i][nSquare % grid_size], mesh[i + 1][(nSquare % grid_size) + 1],
                          color=nSquare + 1, thickness=-1)
            nSquare += 1

    return squares


def compute_stars(image, grid_size, thresh_brightness, squares):
    """
    Create an image with the stars detected and compute the number of stars detected in each square of the grid
    :param image: array, preprocessed image of size 480 x 480 pixels to apply the cloud cover estimation on
    :param grid_size:  int, number of squares on a grid line (must be an integer divisor of 420)
    :param thresh_brightness: int, the minimum brightness of the pixel in order to be considered as a star (between 0
    and 255)
    :param squares: array, 480 x 480 pixels image with a 420 x 420 pixels grid where each square of the grid has a different value
    :return: array, image where stars has 255 value and the rest has a 0 value
             list, number of stars detected in each square of the grid
    """

    [rows, cols, channels] = image.shape
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_stars = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Image with only the detected stars

    nb_stars = [0 for _ in range(grid_size ** 2)]  # Number of stars in each square

    for i in range(0, rows - 1):
        for j in range(0, cols - 1):

            if img_stars[i, j] <= thresh_brightness or is_star(img, [i, j]) == False:
                img_stars[i, j] = 0

            elif img_stars[i, j] > thresh_brightness and is_star(img, [i, j]) == True:
                img_stars[i, j] = 255
                nb_stars[int(squares[i][j]) - 1] += 1

            else:
                img_stars[i, j] = 0

    return img_stars, nb_stars


def compute_cloudy_cells(img_stars, nb_stars, squares, thresh_cell):
    """
    Determine if a cell is considered as a cloud cell or not according to the number of stars previously detected, and
    segment the image according to it
    :param img_stars: array, image where stars has 255 value and the rest has a 0 value
    :param nb_stars: list, number of stars detected in each square of the grid
    :param squares: array, 480 x 480 pixels image with a 420 x 420 pixels grid where each square of the grid has a different value
    :param thresh_cell: int, the number of stars needed at least to be considered as clear
    :return: array, segmented image between clouds (255 value) and sky
    """
    [rows, cols] = img_stars.shape

    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            if nb_stars[int(squares[i][j]) - 1] < thresh_cell:
                img_stars[i, j] = 255

    img_stars = masked_image(img_stars, cmask([240, 240], 210, img_stars))

    return img_stars


def draw_mesh(image, grid_size):
    """
    Draw the grid on the image according to the grid size (number of squares)
    :param image: array, Image to draw the grid on
    :param grid_size: int, number of squares on a grid line (must be an integer divisor of 420)
    :return: None
    """

    incr = int(420 / grid_size)
    list = [30 + incr * i for i in range(0, grid_size + 1)]
    for coord in list:
        cv2.line(image, [coord, 30], [coord, 450], color=(255, 255, 255))
        cv2.line(image, [30, coord], [450, coord], color=(255, 255, 255))


def estimation_stars(image,grid_size=10, thresh_brightness=20, thresh_cell=2):
    """
    Main function to use in order to estimate the cloud cover of an image by using the stars detection method

    :param image: array, preprocessed image of size 480 x 480 pixels to apply the cloud cover estimation on
    :param grid_size: int, the grid has a shape of grid_size * grid_size squares (must be an integral divisor of 420)
    :param thresh_brightness: int, the minimum brightness of the pixel in order to be considered as a star (between 0
    and 255)
    :param thresh_cell: int, the number of stars needed at least to be considered as clear
    :return: (float, array), percentage of the image covered by clouds and the segmented image between clouds (255 value)
    and sky
    """

    print("Parameters of stars based estimation: Grid Size = ", grid_size, ", Detection star threshold = ", thresh_brightness,
          ", Nb of stars per cell threshold = ", thresh_cell)

    # Creating the mesh
    squares = create_mesh(grid_size)

    # Computing the stars
    img_stars, nb_stars = compute_stars(image, grid_size, thresh_brightness, squares)

    # Correction to apply to the final CCR because stars are 255 pixel value counted as clouds
    CCI_only_stars = CCR(img_stars)

    # Computing the clear and the cloudy cells
    img_stars = compute_cloudy_cells(img_stars, nb_stars, squares, thresh_cell)

    # Compute the Cloud Cover Index from the image (with the correction)
    CCI = CCR(img_stars) - CCI_only_stars


    # Draw the mesh grid on the image
    # draw_mesh(img_stars,grid_size)
    # img_stars = masked_image(img_stars, cmask([240, 240], 210, img_stars))

    return round(CCI, 2), img_stars


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.io import imread

    # Testing and example of the is_star function
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    simple_image = np.zeros((3, 3))
    simple_image[1, 1] = 255
    simple_image[0, 0] = 100
    simple_image[0, 1] = 150
    plt.subplot(131)
    plt.imshow(simple_image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(cmap="gray", fraction=0.046, pad=0.04)
    plt.title(str(is_star(simple_image, (1, 1))))

    simple_image = np.zeros((3, 3));
    simple_image[1, 1] = 255;
    simple_image[0, 0] = 255;
    simple_image[0, 1] = 150
    plt.subplot(132)
    plt.imshow(simple_image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(cmap="gray", fraction=0.046, pad=0.04)
    plt.title(str(is_star(simple_image, (1, 1))))

    simple_image = 180 * np.ones((3, 3));
    simple_image[1, 1] = 255
    plt.subplot(133)
    plt.imshow(simple_image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(cmap="gray", fraction=0.046, pad=0.04)
    plt.title(str(is_star(simple_image, (1, 1))))

    plt.suptitle("Result of star detection of the center pixel is", y=0.75)
    plt.show()

    # Mesh example with size 10
    squares = create_mesh(10)
    plt.imshow(squares)
    plt.colorbar(label="Square number in the mesh")
    plt.title("Mesh with the squares")
    plt.show()

    # Detection of stars example
    img_path = "test_images/clear/LYR-Sony-311219_012338-ql.jpg"
    img = imread(img_path)[:-24]
    img_stars = compute_stars(img, 10, 20, squares)[0]

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(img_stars, cmap="gray")
    plt.title("Detected stars")
    plt.show()

    plt.subplot(121)
    plt.imshow(img)
    plt.xlim(100, 150)
    plt.ylim(300, 350)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(img_stars, cmap="gray")
    plt.xlim(100, 150)
    plt.ylim(300, 350)
    plt.title("Detected stars")
    plt.show()

    # Grid drawing
    img_path = "test_images/intermediate/LYR-Sony-150220_183648-ql.jpg"
    img = imread(img_path)[:-24]
    img_grid = imread(img_path)[:-24]
    draw_mesh(img_grid, 10)
    img_grid = masked_image(img_grid, cmask([240, 240], 210, img_grid))

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(img_grid, cmap="gray")
    plt.title("Image with the grid")
    plt.show()

    # Automatic Cloudiness Estimation from stars detection and mesh
    img_path = "test_images/intermediate/LYR-Sony-150220_183648-ql.jpg"
    img = imread(img_path)[:-24]
    CCI, img_stars = estimation_stars(img)

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(122)
    plt.imshow(img_stars, cmap="gray")
    plt.title("Image with cloud estimation \n " + "Cloud cover = " + str(int(CCI)) + "%")
    plt.show()
