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
    # #Example of a cloud cover estimation
    #
    # folder = "test_images/intermediate/"
    # #img_name = "LYR-Sony-150220_183648-ql.jpg"
    # #img_name = "LYR-Sony-080220_003742-ql.jpg"
    # #img_name = "LYR-Sony-080220_083852-ql.jpg"
    # img_name = "LYR-Sony-20171126_090010.jpg"
    #
    # img_path = folder + img_name
    #
    # #date = ("20" + img_name[13:15] + "-" + img_name[11:13] + "-" + img_name[9:11] + "T" + img_name[16:18] + ":" +
    # #        img_name[18:20] + ":" + img_name[20:22])
    #
    # date = (img_name[9:13] + "-" + img_name[13:15] + "-" + img_name[15:17] + "T" + img_name[18:20] + ":" +
    #                       img_name[20:22] + ":" + img_name[22:24])
    #
    # img = imread(img_path)
    #
    # results = cloud_cover_estimation(img, date)
    #
    # plot_cloud_cover(results)

    ##### Airport validation
    import os
    import pandas as pd
    from matplotlib.ticker import MaxNLocator

    DF = pd.read_csv(
        "/Users/alessandro/Desktop/University Center in Svalbard/ImageClassification/OneHourResolution/airport_2017_2018.csv")
    DF2 = pd.read_csv(
        "/Users/alessandro/Desktop/University Center in Svalbard/ImageClassification/OneHourResolution/KHO_2017_2018.csv")
    #print(DF['Time'][0])
    #print(DF[DF['Time'] == '04.01.2018 01:00'])
    # print(DF2)
    print(DF)
    # DF = DF.set_index(pd.to_datetime(DF['Time']))
    # DF2 = DF2.set_index(pd.to_datetime(DF['Time']))

    okta_estimation = []
    okta_airport = []
    times = []

    DF['Estimation'] = pd.Series(dtype='int')

    plt.plot(DF["Cloud cover"], "g--", label="Airport")
    plt.plot(DF2["Cloud cover"], "g-", label="KHO")
    plt.ylabel("Cloud Cover in okta")
    plt.legend()
    plt.show()

    input_dir = '/Users/alessandro/Desktop/University Center in Svalbard/ImageClassification/OneHourResolution/airport_2017_2018/'
    for file in os.listdir(os.path.join(input_dir)):
        if file != ".DS_Store":

            img_path = os.path.join(input_dir, file)

            #img = imread(img_path)
            #print(file)

            date = (file[9:13] + "-" + file[13:15] + "-" + file[15:17] + "T" + file[18:20] + ":" +
                     file[20:22] + ":" + file[22:24])

            print("##################",date,"##################")

            hour = str(int(date[11:13])+1)
            if len(hour) == 1:
                hour = '0' + hour
            date_airport = date[8:10] + '.' + date[5:7] + '.' + date[:4] + " " + hour + date[13:16]

            #print(date_airport)
            #print(DF[DF['Time'] == date_airport]['Cloud cover'])
            #print(date_airport)
            #print(DF[DF['Time'] == date_airport]['Cloud cover'])
            okta_airport = int(DF[DF['Time'] == date_airport]['Cloud cover'].iloc[0])
            okta_kho = int(DF2[DF2['Time'] == date_airport]['Cloud cover'].iloc[0])
            #print(okta)



            img = imread(img_path)

            results = cloud_cover_estimation(img,date)

            DF.loc[DF['Time'] == date_airport, 'Estimation'] = int(round(results[2]/12.5,0))

            plot_cloud_cover(results,
                             path_save="/Users/alessandro/Desktop/University Center in Svalbard/ImageClassification/OneHourResolution/CloudCover_airport_2017_2018/" + file
                             ,airport_okta=okta_airport,
                             kho_okta=okta_kho)

            # plot_cloud_cover(results,
            #                  path_save="/Users/alessandro/Desktop/University Center in Svalbard/ImageClassification/OneHourResolution/CloudCover_airport_2017_2018/" + file
            #                  , airport_okta=okta_airport)

    print(DF)

    DF.to_csv("/Users/alessandro/Desktop/University Center in Svalbard/ImageClassification/OneHourResolution/airport_2017_2018_estimation.csv")
