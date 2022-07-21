import os
import time

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_ndwi(img_dict, image_name):
    ndwi = normalized_difference(img_dict, 'B3', 'B5')

    # checking the images
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # plt.imshow(ndwi, cmap='Blues')
    image = plt.imshow(ndwi, cmap='jet_r')
    # image = plt.imshow(ndwi, cmap='BuGn_r')
    # from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list('rg', ["g", "w", "b"], N=256)
    # image = plt.imshow(ndwi, cmap=cmap)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(image, cax=cax)
    # cbar.set_label('Color', rotation=270, labelpad=25)
    cbar.set_label('NDWI')

    fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close(fig)
    # plt.show()
    # image = Image.fromarray(np.uint8(ndwi * 255))
    # image.show()
    # image.save(image_name)


def list_folders(root_folder):
    return [f.name for f in os.scandir(root_folder) if f.is_dir()]


def get_date_from_folder_name(folder_name):

    return folder_name[17:25]


def normalized_difference(img, b1, b2, eps=0.0001):
    band1 = np.where((img[b1] == 0) & (img[b2] == 0), np.nan, img[b1])
    band2 = np.where((img[b1] == 0) & (img[b2] == 0), np.nan, img[b2])

    return (band1 - band2) / (band1 + band2)


def load_images(img_folder, prefix, bands):

    image_dict = {}

    # UTM coordinates (WGS84)
    # Z: 33V, E: 574027.604, N: 6637469.071
    # Lat: 59.868176 N, Lon: 16.321993 E
    left = 570185.0
    bottom = 6634685.0
    right = 580215.0
    top = 6641815.0

    # Top left: 59.907861 N, 16.254861 E
    # Top right: 59.906033 N, 16.43413 E
    # bottom right: 59.842029 N, 16.431376 E
    # bottom left: 59.843853 N, 16.252451 E

    for band in bands:
        file_name = img_folder + prefix + band + '.TIF'
        ds = rasterio.open(file_name)
        # image_dict.update({band: ds.read(1)})
        image_dict.update({band: ds.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, ds.transform))})

    return image_dict


def plot_water_mask(img_dict, image_name):
    ndwi = normalized_difference(img_dict, 'B3', 'B5')

    water_mask = ndwi > -0.0
    water_ratio = water_mask.sum() / (img_dict['B3'].shape[0] * img_dict['B3'].shape[1])
    print('Water ratio', water_ratio)
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(water_mask)
    fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close(fig)
    # plt.show()

    # image = Image.fromarray(np.uint8(water_mask * 255))
    # image.show()
    # image.save(image_name)


def full_cycle():

    landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/'
    folders_list = list_folders(landsat_folder)
    folders_list.remove('tars')
    print(folders_list)

    for folder in folders_list:

        # img_prefix = 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_'
        img_folder = landsat_folder + folder + '/'
        img_prefix = folder + '_SR_'
        print(img_prefix)
        image_date = get_date_from_folder_name(img_prefix)
        print('Date', image_date)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        img_dict = load_images(img_folder, img_prefix, bands)
        plot_ndwi(img_dict, f'/tmp/ndwi/ndwi-{image_date}.png')
        plot_water_mask(img_dict, f'/tmp/water_mask/water_mask-{image_date}.png')
        print('---------------------------------------\n')


def main():
    landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20190827_20200826_02_T1/'
    img_prefix = 'LC08_L2SP_193018_20190827_20200826_02_T1_SR_'
    landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20200322_20200822_02_T1/'
    img_prefix = 'LC08_L2SP_193018_20200322_20200822_02_T1_SR_'

    output_folder = '/tmp/study_case'
    image_date = get_date_from_folder_name(img_prefix)
    print('Date', image_date)
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    img_dict = load_images(landsat_folder, img_prefix, bands)

    # plot_ndvi_ndwi_examples(img_dict)
    plot_ndwi(img_dict, f'{output_folder}/ndwi-{image_date}.png')
    plot_water_mask(img_dict, f'{output_folder}/water_mask-{image_date}.png')


start = time.time()
# main()
full_cycle()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))