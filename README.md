#  Cloud cover estimation from KHO nighttime images with Python

## Context

The Kjell Henriksen Observatory (KHO) is running a Sony A7S all-sky camera 
which is taking pictures every 6 seconds. A lot of photographs are 
taken and for many of them, the observation of the sky is difficult 
due to the presence of clouds. The aim of this project is to estimate 
cloud cover from a  all-sky picture taken at KHO. 

## General methodology

A Support Vector Machine model has been trained, using a dataset created
with the Aurora Eurotech Cloud Sensor installed at KHO and owned by the
University College London, in order to classify images in three
categories: Full clear, Full cloudy and Intermediate. The first step of the algorithm
is to run the model on the image, if the result is Fully clear or Fully cloudy,
the algorithm ends and 100% or 0% of cloudiness is the result. If the result of the
model prediction is Intermediate, the detection of the Moon is realized using the time
and the position. Then, according to the presence of the Moon or not on the image, various image segmentation
algorithm are computed to give an estimation of the cloudiness. 

## How to use the automatic cloudiness estimation 

The Python-function for estimating the cloud cover from an image is

```
 (image, imageProc, cloud_coverage, comment_coverage) = ...
    cloud_cover_estimation(image, date)
```
or
```
 results = cloud_cover_estimation(image, date)
```
and then the results can be used by the plot function
```
 plot_cloud_cover(results)
```

| Input parameter | Description                                                                    | Type     |
|-----------------|--------------------------------------------------------------------------------|----------|
| `image`         | The image to estimate cloudiness                                               | *array*  |
| `date`          | Date and time <br/>Year-Month-DayTHour:Minute:Second (example 2020-02-08T20:48:06) | *string* |

| Output parameter   | Description                                    | Type     |
|--------------------|------------------------------------------------|----------|
| `image`            | The original image used to estimate cloudiness | *array*  |
| `imageProc`        | The image segmented between clouds and sky     | *array*  |
| `cloud_coverage`   | The cloud cover of the image (in %)            | *float*  |
| `comment_coverage` | Comment about the coverage and the Moon        | *string* |



