"""
A script that contains the Automatic Clouds Estimation from an all-sky image at KHO main function and plot function

Author: Alessandro Marocco
Email: alessandro.marocco@ens.psl.eu
Date: July 2024
"""

# Import the required modules
import pickle
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import os

from image_tools import preprocessing_image, cmask, compute_brightness

from moon_detection import get_moon_position

from stars_estimation import estimation_stars

from colors_estimation import estimation_RBR


def cloud_cover_estimation(image, date=None, name_model="model3_96px.p"):
    """
    Cloud coverage estimation from all-sky image by using an SVC classifier model, moon detection and image algorithms
    :param image: rgb image
    :param date: str, Year-Month-DayTHour:Minute:Second (example 2020-02-08T20:48:06) (mandatory if intermediate case)
    :return:
    """
    imageProc = preprocessing_image(image)

    MODEL = pickle.load(open(name_model, "rb"))  # Importation of the trained model

    prediction = MODEL.predict([resize(imageProc, (96, 96)).flatten()])

    cloud_coverage, comment_coverage = None, None

    # Fully clear case
    if prediction == [0]:
        cloud_coverage = 0
        comment_coverage = "Fully clear"

    # Fully cloudy case
    if prediction == [1]:
        cloud_coverage = 100
        comment_coverage = "Fully cloudy"

        imageProc = cmask([240, 240], 210, imageProc)

    # Intermediate case
    if prediction == [2]:

        pos = get_moon_position(date)
        print("Moon position: ", pos)

        if pos == False:
            comment_coverage = "Intermediate - No Moon"

            # cloud_coverage, imageProc = apply_no_moon(image) #stars peak, otsu binarization
            cloud_coverage, imageProc = estimation_stars(imageProc)
            # cloud_coverage, imageProc = estimation_Otsu(imageProc)

        else:
            comment_coverage = "Intermediate - Moon"

            moon_mask = cmask(pos, 80, imageProc)
            # imageProc = masked_image(imageProc, moon_mask, fill=0, color=255)
            # imageProc = masked_image(imageProc,cmask([240, 240], 210, imageProc))

            brightness = compute_brightness(imageProc)

            if brightness > 170:
                cloud_coverage, imageProc = estimation_RBR(imageProc, moon_mask)  # Red Blue Ratio?
                # cloud_coverage, imageProc = estimation_Otsu(imageProc)

            else:
                cloud_coverage, imageProc = estimation_stars(imageProc)

    return (image, imageProc, cloud_coverage, comment_coverage)


def plot_cloud_cover(results, path_save=None, airport_okta=None, kho_okta=None):
    """
    Plot the results of the cloud cover estimation function showing the original image and the segmentation between sky
    and clouds with the okta value. The option to save the plot can be chosen by specifying a path to save. The okta from
    the airport and/or from kho can be display as well.
    :param results: tuple,
    :param path_save: string,
    :param airport_okta: int,
    :param kho_okta: int,
    :return:
    """

    image, imageProc, cloud_coverage, comment_coverage = results[0], results[1], results[2], results[3]

    px = 1 / plt.rcParams['figure.dpi']
    plt.figure(figsize=(1500 * px, 800 * px))

    plt.subplot(121)
    plt.imshow(image)
    if airport_okta != None:
        plt.title("Svalbard Lufthavn: " + str(airport_okta) + "/8")
    if airport_okta != None and kho_okta != None:
        plt.title("Svalbard Lufthavn: " + str(airport_okta) + "/8 ---> " + "KHO: " + str(kho_okta) + "/8")

    plt.subplot(122)
    plt.imshow(imageProc, cmap="gray")
    plt.title(comment_coverage + "\n" + "Cloud Cover Index = " + str(cloud_coverage) + '%' + " -> " + str(
        int(round(cloud_coverage / 12.5, 0))) + "/8")

    if path_save != None:
        plt.savefig(path_save)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":

    # Example of cloud cover estimation
    
    folder = "test_images"

    for root, folders, files in os.walk(folder):

        for img_name in files:

            img_path = root + '/' + img_name
            
            date = ("20" + img_name[13:15] + "-" + img_name[11:13] + "-" + img_name[9:11] + "T" + img_name[16:18] + ":" +
                    img_name[18:20] + ":" + img_name[20:22])
    
            img = imread(img_path)
    
            results = cloud_cover_estimation(img, date)
    
            plot_cloud_cover(results)

    # For only one cloud cover estimation

    folder = "test_images/intermediate/"
    
    img_name = "LYR-Sony-080220_003742-ql.jpg"

    img_path = folder + img_name

    date = ("20" + img_name[13:15] + "-" + img_name[11:13] + "-" + img_name[9:11] + "T" + img_name[16:18] + ":" +
                    img_name[18:20] + ":" + img_name[20:22])
    
    img = imread(img_path)
    
    results = cloud_cover_estimation(img, date)
    
    plot_cloud_cover(results)



    