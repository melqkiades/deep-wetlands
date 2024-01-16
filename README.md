# Deep Wetlands
Repository for the Deep Wetlands project in which we detect changes in water extension of wetlands my using multispectral and SAR images.

See more information at <https://www.digitalfutures.kth.se/research/postdoc-fellowships/deep-wetlands/>.

## Journal Publication

If you use our model please cite our publication.

Francisco J. Peña, Clara Hübinger, Amir H. Payberah, Fernando Jaramillo, _"DeepAqua: Semantic segmentation of wetland water surfaces with SAR imagery using deep neural networks without manually annotated data"_, International Journal of Applied Earth Observation and Geoinformation, Volume 126, 2024, 103624, ISSN 1569-8432,

https://www.sciencedirect.com/science/article/pii/S156984322300448X

https://doi.org/10.1016/j.jag.2023.103624


## Download Pre-trained Models

You can download the pre-trained models and the test data from the links below.

To make things faster, after you click on the download link, click on the three dots and then click organize. You can then select a folder in your Google Drive. For this tutorial, I have created a folder called `DeepAqua-Notebook`. Make sure your data is stored in the folder `/content/drive/MyDrive/DeepAqua-Notebook/`.

Our pre-trained models are optimized for specific years due to occasional adjustments made to the Sentinel-1 sensor. These adjustments result in variations in the range of values (histograms) in the TIFF images. Consequently, if you aim to segment water from Sentinel-1 imagery from years not covered by our pre-trained models, you'll need to train a model yourself using data from those specific years.

### Small Models

#### Years 2018-2019
https://drive.google.com/file/d/1gRj98jWhvRSeLoAzcNzwM6Gi9I8K__0-

#### Years 2020-2022
https://drive.google.com/file/d/1N7ca5fKTGdazw7n8ALCYH6m3nsG2SUII

### Large Models

#### Years 2018-2019
https://drive.google.com/file/d/11CFnSUrKsjvTo4JqcFxKXP7RwCmwSUze

#### Years 2020-2022
https://drive.google.com/file/d/1fJeg6hPMORZoNkcUC-zh7XlaPL6Bj-FA

## Tutorial on Using the Pre-trained Models

The following Google Colab notebook illustrates how to use the pre-trained models.

https://colab.research.google.com/drive/1tY3Q8QxZdOG2pFUh9XS17jPEy_WkBOAl?authuser=2#scrollTo=pNBvcrnYL0_N

## Questions

If you have any questions you can always contact me at:

francisco (dot) pena (at) natgeo (dot) su (dot) se
