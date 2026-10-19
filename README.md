#  Cloud cover estimation from KHO nighttime images with Python

## Context

The Kjell Henriksen Observatory (KHO), Svalbard, is running a Sony A7S all-sky
camera which is taking pictures every 6 seconds during the polar night. A lot of
photographs are taken and for many of them, the observation of the sky is
difficult due to the presence of clouds. The aim of this project is to estimate
cloud cover from an all-sky picture taken at KHO.

This work was carried out during an M1 internship at the Arctic Geophysics
department of The University Centre in Svalbard (UNIS), from February to July
2024.

## General methodology

A Support Vector Machine model has been trained, using a dataset created with
the Aurora Eurotech Cloud Sensor installed at KHO and owned by University
College London, in order to classify images in three categories: Fully clear,
Fully cloudy and Intermediate. The first step of the algorithm is to run the
model on the image. If the result is Fully clear or Fully cloudy, the algorithm
ends and 0 % or 100 % of cloudiness is the result. If the result of the model
prediction is Intermediate, the position of the Moon is computed from the date
and time. Then, depending on whether the Moon is in the sky, various image
segmentation algorithms are used to estimate the cloudiness.

<img width="1400" height="480" alt="Image" src="https://github.com/user-attachments/assets/4422738f-2dfe-4359-b24d-27b7ab6523b6" />

## Performance

Over the winter 2017–2018 (940 observations), the estimates were compared with
the cloud cover observed at the Svalbard Airport weather station, corrected for
KHO. The mean difference is 0.69 okta, and 86 % of the estimates are within
1 okta of the observation. Details on the training data and the evaluation are
given on the [Hugging Face model card](https://huggingface.co/alessandromarocco/KHO-cloudiness-svm).

## Installation

```bash
git clone https://github.com/alessandromarocco/KHO-AutomaticCloudinessEstimation.git
cd KHO-AutomaticCloudinessEstimation
pip install -r requirements.txt
```

## Download the model

The Support Vector Machine model (~328 MB) is too large for GitHub and is hosted
on Hugging Face: https://huggingface.co/alessandromarocco/KHO-cloudiness-svm

Download it into the root of this repository with:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="alessandromarocco/KHO-cloudiness-svm",
    filename="model3_96px.p",
    local_dir=".",
)
```

The model must be loaded with scikit-learn 1.4.2 (pinned in `requirements.txt`).

## Quick start

Once the model is downloaded, run the example on the images of `test_images/`:

```bash
python automatic_cloud_estimation.py
```

For each image, a figure shows the original image and the segmentation between
clouds and sky, with the cloud cover in % and in okta.

## How to use the automatic cloudiness estimation

The Python function for estimating the cloud cover from an image is

```python
(image, imageProc, cloud_coverage, comment_coverage) = cloud_cover_estimation(image, date)
```
or
```python
results = cloud_cover_estimation(image, date)
```
and then the results can be used by the plot function
```python
plot_cloud_cover(results)
```

| Input parameter | Description | Type |
|---|---|---|
| `image` | The image to estimate cloudiness | *array* |
| `date` | Date and time, `YYYY-MM-DDTHH:MM:SS` (example `2020-02-08T20:48:06`). Mandatory for the Intermediate case, where it is used to locate the Moon | *string* |

| Output parameter | Description | Type |
|---|---|---|
| `image` | The original image used to estimate cloudiness | *array* |
| `imageProc` | The image segmented between clouds and sky | *array* |
| `cloud_coverage` | The cloud cover of the image (in %) | *float* |
| `comment_coverage` | Comment about the coverage and the Moon | *string* |

`plot_cloud_cover` can also save the figure (`path_save`) and display the okta
observed at the Svalbard Airport and/or at KHO (`airport_okta`, `kho_okta`) for
comparison.

### Expected input images

- KHO quick-look images of 480 × 480 px, or 480 × 504 px (the information band
  at the bottom is then removed automatically).
- Magnetic North at the top of the image, as delivered by KHO.
- The code is designed for this camera only. Images from another camera or
  site would need a new training of the model and new image parameters.

The example script reads the date from the KHO file name,
`LYR-Sony-DDMMYY_HHMMSS-ql.jpg`. For example, `LYR-Sony-080220_003742-ql.jpg`
gives `2020-02-08T00:37:42`.

## Repository structure

| File / folder | Content |
|---|---|
| `automatic_cloud_estimation.py` | Main function `cloud_cover_estimation`, plot function `plot_cloud_cover` and example |
| `image_tools.py` | Preprocessing, circular masks, brightness and cloud cover ratio |
| `moon_detection.py` | Position of the Moon on the image from the date, with Astropy |
| `stars_estimation.py` | Cloud cover estimation by star detection |
| `colors_estimation.py` | Cloud cover estimation from the red and blue channels (used with a bright Moon) |
| `otsu_estimation.py` | Otsu thresholding, tested but not used in the current pipeline |
| `test_images/` | Example images in three folders: `clear`, `clouds` and `intermediate` |
| `requirements.txt` | Python dependencies |

## License

This project is licensed under the GNU General Public License v3.0 — see
[LICENSE](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@software{marocco2024kho,
  author = {Marocco, Alessandro},
  title  = {Cloud cover estimation from KHO nighttime images},
  year   = {2024},
  url    = {https://github.com/alessandromarocco/KHO-AutomaticCloudinessEstimation}
}
```
