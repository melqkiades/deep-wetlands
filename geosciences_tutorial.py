import collections
import os
import time
from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.warp import reproject


def open_example_image():

    landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20210512_20210524_02_T1/'

    ds = rasterio.open(landsat_folder + 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_B4.TIF')
    img = ds.read(1)

    print(img.shape)
    print(img.min(), img.max())
    plt.imshow(img)
    plt.show()


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


def display_rgb(img, image_name, b_r, b_g, b_b, alpha=1., figsize=(10, 10)):
    rgb = np.stack([img[b_r], img[b_g], img[b_b]], axis=-1)
    rgb = rgb/rgb.max() * alpha
    fig = plt.figure(figsize=figsize)
    plt.imshow(rgb)
    # plt.show()
    fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close(fig)

    # image = Image.fromarray(np.uint8(rgb * 255))
    # image.show()
    # image.save(image_name)


def normalized_difference(img, b1, b2, eps=0.0001):
    band1 = np.where((img[b1] == 0) & (img[b2] == 0), np.nan, img[b1])
    band2 = np.where((img[b1] == 0) & (img[b2] == 0), np.nan, img[b2])

    return (band1 - band2) / (band1 + band2)


def plot_ndvi_ndwi_examples(img_dict):
    ndvi = normalized_difference(img_dict, 'B5', 'B4')

    # NDWI = (Band 3 – Band 5) / (Band 3 + Band 5)
    mndwi = normalized_difference(img_dict, 'B3', 'B5')

    # checking the images
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(ndvi, cmap='Greens')
    ax[1].imshow(mndwi, cmap='Blues')
    plt.show()

    water_mask = mndwi > -0.0
    plt.figure(figsize=(7, 7))
    plt.imshow(water_mask)
    plt.show()


def plot_ndvi(img_dict, image_name):
    ndvi = normalized_difference(img_dict, 'B5', 'B4')

    # checking the images
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # plt.imshow(ndvi, cmap='Greens')
    # image = plt.imshow(ndvi, cmap='jet')
    image = plt.imshow(ndvi, cmap='RdYlGn')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(image, cax=cax)
    # cbar.set_label('Color', rotation=270, labelpad=25)
    cbar.set_label('NDVI')
    fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close(fig)
    # plt.show()
    # image = Image.fromarray(np.uint8(ndvi * 255))
    # image.show()
    # image.save(image_name)


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

    return water_ratio


def zoom_image(img_dict, highlight_points):
    rgb = np.stack([img_dict['B4'], img_dict['B3'], img_dict['B2']], axis=-1)

    # normalize the values
    rgb = rgb / rgb.max() * 2

    # display the image with a slightly increased figure size
    plt.figure(figsize=(7, 7))

    for x, y in highlight_points:
        rgb[x, y] = 1.0
    plt.imshow(rgb[4800:5000, 1900:2100, 0:3])

    plt.show()


def mask_non_water_pixels(img_dict):
    mndwi = normalized_difference(img_dict, 'B3', 'B5')
    water_mask = mndwi > -0.0

    # create the normalized rgb cube
    rgb = np.stack([img_dict['B4'], img_dict['B3'], img_dict['B2']], axis=-1)
    rgb = rgb/rgb.max() * 2

    # calc the MNDWI index
    # mndwi = normalized_difference(img, 'B3', 'B5')

    # get a Boolean water mask
    # water_mask = mndwi > 0.0

    # Assign 0 to values outside the mask
    rgb[~water_mask] = 0

    # display result
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb*2)
    plt.show()

    # Step 3- Overlay the water mask in RGB

    rgb = np.stack([img_dict['B4'], img_dict['B3'], img_dict['B2']], axis=-1)
    rgb = rgb / rgb.max() * 2

    # Assign values R=0.1, G=0.1 and B=0.9 to the water pixels
    rgb[water_mask] = [0.1, 0.1, 0.9]

    # display result
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb)
    plt.show()


def stack_img(img, bands):
    # create a list of the band's arrays
    bands_arrays = [img[band] for band in bands]
    return np.stack(bands_arrays, axis=-1)


def show_spectral_signature(img_dict, points_list):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    cube = stack_img(img_dict, bands)

    for x, y in points_list:
        plt.plot(bands, cube[x, y])

    plt.show()


# def show_water_spectral_signatures(img_dict):
#     mndwi = normalized_difference(img_dict, 'B3', 'B5')
#     water_mask = mndwi > -0.0
#     bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
#     cube = stack_img(img_dict, bands)
#     water_pts = cube[water_mask] / 10000
#     wls = [0.44, 0.47, 0.56, 0.655, 0.865, 1.61, 2.2]
#     mean = water_pts.mean(axis=0)
#     std = water_pts.std(axis=0)
#     plt.figure(figsize=(10, 6))
#     plt.plot(wls, mean, label='water')
#     plt.fill_between(wls, mean - 0.5 * std, mean + 0.5 * std, color='blue', alpha=0.1)
#     plt.legend()
#     plt.show()


def show_water_spectral_signatures(img_dict):
    mndwi = normalized_difference(img_dict, 'B3', 'B5')
    water_mask = mndwi > -0.0
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    cube = stack_img(img_dict, bands)

    # get the water pixels (remember it has 7 channels)
    water_pts = cube[water_mask] / 10000

    # create the wavelenghts list
    wls = [0.44, 0.47, 0.56, 0.655, 0.865, 1.61, 2.2]

    # get mean and std vectors, by reducint in the first axis
    mean = water_pts.mean(axis=0)
    std = water_pts.std(axis=0)

    # plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wls, mean, label='water')
    plt.fill_between(wls, mean - 0.5 * std, mean + 0.5 * std, color='blue', alpha=0.1)
    plt.xlabel('Wavelenght (nm)')
    plt.ylabel('Reflectance (sr-1)')
    plt.legend()
    plt.show()


def show_vegetation_spectral_signatures(img_dict):
    mndwi = normalized_difference(img_dict, 'B3', 'B5')
    water_mask = mndwi > -0.0
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    cube = stack_img(img_dict, bands)

    # get the water pixels (remember it has 7 channels)
    water_pts = cube[water_mask] / 10000

    # create the wavelenghts list
    wls = [0.44, 0.47, 0.56, 0.655, 0.865, 1.61, 2.2]

    # get mean and std vectors, by reducint in the first axis
    mean = water_pts.mean(axis=0)
    std = water_pts.std(axis=0)

    ndvi = normalized_difference(img_dict, 'B5', 'B4')
    veg_mask = ndvi > 0.25  # adopting a threshold of 0.5 for the vegetation

    veg_pts = cube[veg_mask] / 10000
    veg_mean = veg_pts.mean(axis=0)
    veg_std = veg_pts.std(axis=0)

    # plot the spectrum
    plt.figure(figsize=(10, 6))

    # plot vegetation
    plt.plot(wls, veg_mean, label='Vegetation', color='green')
    plt.fill_between(wls, veg_mean - veg_std, veg_mean + veg_std, color='green', alpha=0.1)

    # plot water
    plt.plot(wls, mean, label='Water')
    plt.fill_between(wls, mean - 0.5 * std, mean + 0.5 * std, color='blue', alpha=0.1)

    plt.xlabel('Wavelenght (nm)')
    plt.ylabel('Reflectance (sr-1)')
    plt.legend()
    plt.show()


def slice_window_test(img_dict):
    # with rasterio.open("../assets/output/reprojection.tif", crs="EPSG:4326") as src:
    #     width = src.shape[1]
    #     height = src.shape[0]
    #     print(width / 2)
    #     print(height / 2)
    #     print(src.shape)
    #     w = src.read(1, window=rasterio.windows.from_bounds(4.45, 51.48, 4.451, 51.481, src.transform))

    landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20210512_20210524_02_T1/'

    src_7 = rasterio.open(landsat_folder + 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_B7.TIF')
    src_5 = rasterio.open(landsat_folder + 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_B5.TIF')
    src_3 = rasterio.open(landsat_folder + 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_B3.TIF')

    # 'B7', 'B5', 'B3',

    left = 59.8490
    top = 16.4254
    right = 59.9117
    bottom = 16.2550

    # top right
    x_top_right = 1828467.16
    y_top_right = 8380105.05

    # bottom left
    x_bottom_left = 1809498.32
    y_bottom_left = 8366195.86

    # full_image = img_dict['B3']

    # image_window = src.read(1, window=rasterio.windows.from_bounds(x_bottom_left, y_bottom_left, x_top_right, y_top_right, src.transform))
    # image_window = src.read(1, window=rasterio.windows.from_bounds(y_bottom_left, x_bottom_left, y_top_right, x_top_right, src.transform))
    # image_window = src.read(1, window=rasterio.windows.from_bounds(x_bottom_left, y_bottom_left, x_top_right, y_top_right, src.transform))
    # image_window = src.read(1, window=rasterio.windows.from_bounds(x_bottom_left, y_bottom_left, x_top_right, y_top_right, src.transform))
    # # image_window = src.read(1, window=rasterio.windows.from_bounds(4.45, 51.48, 4.451, 51.481, src.transform))

    left = 570185.0
    bottom = 6634685.0
    right = 580215.0
    top = 6641815.0

    image_window_7 = src_7.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, src_7.transform))
    image_window_5 = src_5.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, src_5.transform))
    image_window_3 = src_3.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, src_3.transform))

    #
    rgb = np.stack([image_window_7, image_window_5, image_window_3], axis=-1)
    rgb = rgb / rgb.max() * 2
    # # plt.imshow(img_dict['B3'])
    # matplotlib.use('Qt5Agg')
    # plt.imshow(image_window)
    plt.imshow(rgb)
    plt.savefig('/tmp/flackjson.pdf')
    # plt.show()

    # rgb *= 255
    # print(rgb[0][0])
    # rgb.astype(int)
    # print(rgb)
    print('Max', np.max(rgb), 'Min', np.min(rgb))
    im = Image.fromarray(np.uint8(rgb * 255))
    im.show()
    # image = Image.fromarray(rgb, 'RGB')
    # image.save('/tmp/my.png')
    # image.show()


    img = src_7.read(1)

    print(src_7.transform)
    print('CRS', src_7.crs)
    print('Bounds', src_7.bounds)
    print('Shape', img.shape)
    print(img.min(), img.max())
    # plt.imshow(img)
    # plt.show()

    # with rasterio.open("../assets/output/reprojection.tif", crs="EPSG:4326") as src:
    #     width = src.shape[1]
    #     height = src.shape[0]
    #     print(width / 2)
    #     print(height / 2)
    #     print(src.shape)
    #     w = src.read(1, window=rasterio.windows.from_bounds(4.45, 51.48, 4.451, 51.481, src.transform))


def list_folders(root_folder):
    return [f.name for f in os.scandir(root_folder) if f.is_dir()]


def get_date_from_folder_name(folder_name):

    return folder_name[17:25]


def load_sar_image_asf():

    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190823/S1A_IW_GRDH_1SDV_20190823T162938_20190823T163003_028699_033FA1_9BCC.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20190823t162938-20190823t163003-028699-033fa1-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20190823t162938-20190823t163003-028699-033fa1-002.tiff'
    # file_path = folder + 'crs_001.tif'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190822/S1B_IW_GRDH_1SDV_20190822T163717_20190822T163742_017701_0214D7_2492.SAFE/measurement/'
    # file_path = folder + 's1b-iw-grd-vh-20190822t163717-20190822t163742-017701-0214d7-002.tiff'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190822/S1B_IW_GRDH_1SDV_20190822T052222_20190822T052247_017694_02149F_1F87.SAFE/measurement/'
    # file_path = folder + 's1b-iw-grd-vv-20190822t052222-20190822t052247-017694-02149f-001.tiff'
    # file_path = folder + 's1b-iw-grd-vh-20190822t052222-20190822t052247-017694-02149f-002.tiff'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T052325_20200319T052350_031740_03A919_CF30.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20200319t052325-20200319t052350-031740-03a919-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20200319t052325-20200319t052350-031740-03a919-002.tiff'
    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T163748_20200319T163813_031747_03A952_8BAF.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20200319t163748-20200319t163813-031747-03a952-001.tiff'
    file_path = folder + 's1a-iw-grd-vh-20200319t163748-20200319t163813-031747-03a952-002.tiff'


    # dst = rasterio.open(file_path)
    # print(dst.crs)
    # gcps, gcp_crs = dst.gcps
    # print(gcp_crs)
    #
    # img = dst.read(1)
    # print(img)
    # print('Bounds', dst.bounds)
    # print('Img shape', img.shape)
    # print('Img min, max', img.min(), img.max(), img.mean(), np.median(img), img.std())
    # # print(collections.Counter(map(tuple, img)))
    # print(np.unique(img, return_counts=True))
    # print('ratio = ', 23961751.0/(img.shape[0] * img.shape[1]))

    # img = dst.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, ds.transform))

    # rgb = np.stack([img], axis=-1)
    # rgb = img / img.max() * 1000
    # # rgb = rgb[0:20, 0:20]
    # plt.imshow(rgb, cmap='gray')
    # plt.show()
    # print(rgb)
    # print('RGB min, max', rgb.min(), rgb.max())
    # print('RGB mean', rgb.mean(), np.median(rgb), rgb.std())

    dataset = rasterio.open(file_path)

    print('Bounds:', dataset.bounds)
    print('Coordinate system:', dataset.crs)
    gcps, gcp_crs = dataset.gcps
    print(gcp_crs)
    img = dataset.read(1)
    img = img.astype(float)

    print(type(img), img.shape, img.min(), img.max(), img.mean(), img.std())


    img[img == dataset.nodata] = np.nan  # Convert NoData to NaN
    # img = img[9900:10500, 6400:7100]

    # Alos Palsar coordinates
    img = img[9000:9800, 17500:18500]
    vmin, vmax = np.nanpercentile(img, (5, 95))  # 5-95% stretch

    img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    plt.show()
    # img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    # img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax = plt.gca()
    # img_plt = plt.imshow(img, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
    img_plt = plt.imshow(img, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img_plt, cax=cax)
    # cbar.set_label('Color', rotation=270, labelpad=25)
    cbar.set_label('NDWI')
    # plt.show()
    # plt.savefig('/tmp/sentinel1-vh.pdf')
    plt.savefig('/tmp/sentinel-1.png')

    # Water mask

    # water_mask = img > 60
    # print('Sum', water_mask.sum())
    # print(water_mask)
    # # fig = plt.figure(figsize=(7, 7))
    # # plt.imshow(water_mask, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    # plt.imshow(water_mask,  origin='lower')
    # # fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    # plt.cla()
    # # plt.close(fig)
    # plt.show()

    # print(img.shape)
    # fig = plt.figure()
    # plt.imshow(img)
    # plt.show()

    # from rasterio.plot import show
    # show(dst)


def load_sar_image_asf_alos_palsar():

    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/ALOS-PALSAR/20090813/ALPSRP189311190-H2.2_UA/'
    file_path = folder + 'HH-ALPSRP189311190-H2.2_UA.tif'
    # file_path = folder + 'HV-ALPSRP189311190-H2.2_UA.tif'
    dataset = rasterio.open(file_path)

    print('Bounds:', dataset.bounds)
    print('Coordinate system:', dataset.crs)
    gcps, gcp_crs = dataset.gcps
    print(gcp_crs)
    # img = dataset.read(1)
    left = 570185.0
    bottom = 6634685.0
    right = 580215.0
    top = 6641815.0
    img = dataset.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, dataset.transform))
    img = img.astype(float)

    img[img == dataset.nodata] = np.nan  # Convert NoData to NaN
    # img = img[9900:10500, 6400:7100]
    vmin, vmax = np.nanpercentile(img, (5, 95))  # 5-95% stretch
    img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    # img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    # plt.show()
    plt.savefig('/tmp/alos-palsar-hh.png')



def load_sar_image_gee():
    file_name = '/tmp/S1_test.tif'
    from rasterio.plot import show

    ds = rasterio.open(file_name)
    print(ds.bounds)

    left = 16.252
    bottom = 59.84
    right = 16.434
    top = 59.908

    fig = plt.figure()
    # img = ds.read(1)
    img = ds.read(1, window=rasterio.windows.from_bounds(left, bottom, right, top, ds.transform))

    print(img.shape)
    print(img.min(), img.max())
    plt.imshow(img)
    plt.show()
    # show(rasterio.open(file_name))


def full_cycle():

    landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/'
    folders_list = list_folders(landsat_folder)
    folders_list.remove('tars')
    print(folders_list)

    water_ratios = []

    for folder in folders_list:

        # img_prefix = 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_'
        img_folder = landsat_folder + folder + '/'
        img_prefix = folder + '_SR_'
        print(img_prefix)
        image_date = get_date_from_folder_name(img_prefix)
        print('Date', image_date)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        img_dict = load_images(img_folder, img_prefix, bands)

        # natural color
        display_rgb(img_dict, f'/tmp/rgb/rgb-{image_date}.png', 'B4', 'B3', 'B2', alpha=1.)

        # false color using shortwave infrared B7, near infrared B5 and green.
        display_rgb(img_dict, f'/tmp/nif/nif-{image_date}.png', 'B7', 'B5', 'B3', alpha=2.)

        # plot_ndvi_ndwi_examples(img_dict)
        plot_ndvi(img_dict, f'/tmp/ndvi/ndvi-{image_date}.png')
        plot_ndwi(img_dict, f'/tmp/ndwi/ndwi-{image_date}.png')
        water_ratio = plot_water_mask(img_dict, f'/tmp/water_mask/water_mask-{image_date}.png')
        water_ratios.append(image_date + '-' + str(water_ratio))

    print(water_ratios)




def main():
    # # open_example_image()

    # # landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20210512_20210524_02_T1/'
    # # img_prefix = 'LC08_L2SP_193018_20210512_20210524_02_T1_SR_'
    # landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20190827_20200826_02_T1/'
    # img_prefix = 'LC08_L2SP_193018_20190827_20200826_02_T1_SR_'
    # landsat_folder = '/Users/frape/Projects/DeepWetlands/Datasets/USGS/Landsat/LC08_L2SP_193018_20200322_20200822_02_T1/'
    # img_prefix = 'LC08_L2SP_193018_20200322_20200822_02_T1_SR_'
    #
    # output_folder = '/tmp/study_case'
    # image_date = get_date_from_folder_name(img_prefix)
    # print('Date', image_date)
    # bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    # img_dict = load_images(landsat_folder, img_prefix, bands)
    #
    # # natural color
    # display_rgb(img_dict, f'{output_folder}/rgb-{image_date}.png', 'B4', 'B3', 'B2', alpha=1.)
    #
    # # false color using shortwave infrared B7, near infrared B5 and green.
    # display_rgb(img_dict, f'{output_folder}/nif-{image_date}.png', 'B7', 'B5', 'B3', alpha=2.)
    #
    # # plot_ndvi_ndwi_examples(img_dict)
    # plot_ndvi(img_dict, f'{output_folder}/ndvi-{image_date}.png')
    # plot_ndwi(img_dict, f'{output_folder}/ndwi-{image_date}.png')
    # plot_water_mask(img_dict, f'{output_folder}/water_mask-{image_date}.png')
    #
    # # highlight_points = [(4900, 2000), (4900, 1950), (4950, 2050)]
    # #
    # # # zoom_image(img_dict, highlight_points)
    # # mask_non_water_pixels(img_dict)
    # # show_spectral_signature(img_dict, highlight_points)
    # # show_water_spectral_signatures(img_dict)
    # # show_vegetation_spectral_signatures(img_dict)
    #
    # # slice_window_test(img_dict)
    # # list_folders()

    # full_cycle()
    # load_sar_image_gee()
    load_sar_image_asf()
    # load_sar_image_asf_alos_palsar()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
