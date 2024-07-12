"""
A script to define functions used to detect the presence and position of the Moon on an image from its date and time
using the Astropy package

Author: Alessandro Marocco
Email: alessandro.marocco@ens.psl.eu
Date: July 2024
"""

# Import the required modules
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body, AltAz
from astropy import units as u

import numpy as np


def altaz_to_imagecoordinates(alt, az):
    """
    Convert the Altitude and Azimuth information of an Astropy body to the corresponding coordinates on the image using a linear
    lens deformation of the KHO camera
    :param alt: float, Altitude of the Astropy body
    :param az: float, Azimuth of the Astropy body
    :return: float, float, x and y position of the Astropy body on the image
    """
    # Correction of the azimuth if it exceeds 360 degree
    if az >= 360:
        az = az - 360

    R = 240 # Radius of the area of interest
    r = R * (1 - alt / 90) # Distance from the center of the image to the position of the Moon on the image
    theta = az % 90 # Anticlockwise angle from the North to the position of the Moon on the image

    # Top left quarter of the image
    if 0 <= az < 90:
        x_moon = R - r * np.cos(np.deg2rad(theta))
        y_moon = R - r * np.sin(np.deg2rad(theta))

    # Bottom left quarter of the image
    elif 90 <= az < 180:
        x_moon = R + r * np.sin(np.deg2rad(theta))
        y_moon = R - r * np.cos(np.deg2rad(theta))

    # Bottom right quarter of the image
    elif 180 <= az < 270:
        x_moon = R + r * np.cos(np.deg2rad(theta))
        y_moon = R + r * np.sin(np.deg2rad(theta))

    # Top right quarter of the image
    elif 270 <= az < 360:
        x_moon = R - r * np.sin(np.deg2rad(theta))
        y_moon = R + r * np.cos(np.deg2rad(theta))

    return x_moon, y_moon


def get_moon_position(date):
    """
    Return the moon position on the image using the Astropy package from the date and the location (KHO) using a linear
    lens deformation of the camera at KHO
    :param date: str, time at which position is to be obtained
    :return: x,y: pixel positions of the moon on the all-sky image
    """

    # Define the time and localization of the image
    t = Time(date)

    kho = EarthLocation(lon=16.043000 * u.deg, lat=78.148000 * u.deg, height=520 * u.m)

    # Get the altitude and azimuth of the Moon for the time and localization
    moon = get_body("moon", t)
    altazframe = AltAz(obstime=t, location=kho, pressure=0)
    moonaz = moon.transform_to(altazframe)

    # Case when the Moon is above the horizon (visible on the image) with a margin of 5°
    if moonaz.alt.degree >= -5:
        # Get the position of the Moon in the image, +28.2 degree is to take into account the rotation to the north
        x, y = altaz_to_imagecoordinates(moonaz.alt.degree,
                                         moonaz.az.degree + 28.2)
        return [x, y]

    # Case when the Moon is below the horizon (not visible on the image)
    else:
        return False


if __name__ == "__main__":
    from skimage.io import imread
    from image_tools import cmask, masked_image
    import matplotlib.pyplot as plt

    folder = "test_images/intermediate/"
    img_name = "LYR-Sony-290220_181403-ql.jpg"
    img_path = folder + img_name

    date = ("20" + img_name[13:15] + "-" + img_name[11:13] + "-" + img_name[9:11] + "T" + img_name[16:18] + ":" +
            img_name[18:20] + ":" + img_name[20:22])

    x_moon, y_moon = get_moon_position(date)
    print("Coordinates of the Moon [x,y] on the image of size 480x480: ", [x_moon, y_moon])

    img = imread(img_path)[:-24]
    mask_moon = cmask((x_moon, y_moon), 50, img)
    img_moon = masked_image(img, mask_moon, )

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")

    plt.subplot(122)
    plt.imshow(img_moon)
    plt.title("Moon detection \n Coordinates: [" + str(int(x_moon)) + ";" + str(int(y_moon)) + "]")

    plt.show()
