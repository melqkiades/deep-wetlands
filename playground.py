import os
import time

import cv2
import numpy
import pandas
from PIL import Image
import rasterio as rio
from matplotlib import pyplot as plt
from rasterio.plot import show


def clip_matrix():
    matrix = numpy.array(range(12)).reshape(3, 4)
    print(matrix)
    matrix = numpy.clip(matrix, 3, 7)
    print(matrix)


def count_pixels_ratio():

    # image_path = '/Users/frape/Projects/DeepWetlands/Blog/Water Detection Using NDWI/binary.png'
    image_path = '/Users/frape/Projects/DeepWetlands/Blog/Water Detection Using NDWI/binary_large.png'
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    total_pixels = img.size
    n_white_pix = np.sum(img == 255)
    n_black_pix = np.sum(img == 0)
    print('Number of pixels:', total_pixels)
    print(f'Number of white pixels: {n_white_pix} ({n_white_pix/total_pixels*100:.2f}%)')
    print(f'Number of black pixels: {n_black_pix} ({n_black_pix/total_pixels*100:.2f}%)')
    print('Error:', (total_pixels - n_white_pix - n_black_pix) / total_pixels)

    total_area = 3.929999079856966
    total_area = 21.841035812707247
    water_area = n_white_pix/total_pixels * total_area

    print('Water area (km2):', water_area)


def generate_single_sample_csv():

    sample = {
        # 'ids': 0,
        'indices': 999,
        'split': 'train',
        'max_lat': 47.95622253,
        'max_lon': 2.055749655,
        'min_lat': 47.94274521,
        'min_lon': 2.035627365,
    }

    samples = []

    for counter in range(1572):
        new_sample = sample.copy()
        # index = counter
        # new_sample['ids'] = index
        samples.append(new_sample)

    for counter in range(198):
        new_sample = sample.copy()
        # index = counter + 1572
        # new_sample['ids'] = index
        new_sample['split'] = 'val'
        samples.append(new_sample)

    for counter in range(196):
        new_sample = sample.copy()
        # index = counter + 1572 + 198
        # new_sample['ids'] = index
        new_sample['split'] = 'test'
        samples.append(new_sample)

    data_frame = pandas.DataFrame(samples)
    data_frame.to_csv('/tmp/flacksjon.csv', index_label='ids')


def transform_image_grayscale():

    image_path = '/tmp/999.png'
    img = Image.open(image_path).convert('L')
    img.save('/tmp/greyscale.png')


def resize_image():
    size = (224, 224)
    im = Image.open("/tmp/999.jpeg")
    im.thumbnail(size, Image.ANTIALIAS)
    im.save("/tmp/999-thumbnail.jpeg", "JPEG")

    im = Image.open("/tmp/999.png")
    im.thumbnail(size, Image.ANTIALIAS)
    im.save("/tmp/999-thumbnail.png", "PNG")


def parallel_array_replace():

    arr1 = numpy.array([3, 3, 3, numpy.nan, numpy.nan, numpy.nan])
    arr2 = numpy.array([numpy.nan, 5, 5, numpy.nan, 5, numpy.nan])

    arr1 = numpy.where(numpy.isnan(arr2), arr2, arr1)
    arr2 = numpy.where(numpy.isnan(arr1), arr1, arr2)
    # arr1 = numpy.where(arr2 == 0, arr2, arr1)
    # arr2 = numpy.where(arr1 == 0, arr1, arr2)

    print(arr1)
    print(arr2)


def count_class_labels():

    arr = numpy.array([
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ])

    unique, counts = numpy.unique(arr, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)


def min_max_scale_nan():
    arr = numpy.array([1, 2, 3, numpy.nan, 4])

    array_min, array_max = numpy.nanmin(arr), numpy.nanmax(arr)
    normalized_array = (arr - array_min)/(array_max - array_min)
    print(normalized_array)


def normalize_tiff():
    file_path = '/tmp/Sala kommun-sar-2018-05-01.tif'
    tiff_image = rio.open(file_path)
    numpy_image = tiff_image.read(1)
    print(numpy_image.shape, type(numpy_image), numpy_image.dtype)
    print(numpy_image.min(), numpy_image.max(), numpy_image.mean(), numpy.median(numpy_image))
    array_min, array_max = numpy.nanmin(numpy_image), numpy.nanmax(numpy_image)
    normalized_array = (numpy_image - array_min) / (array_max - array_min)
    normalized_array[numpy.isnan(normalized_array)] = 0
    # normalized_array = (normalized_array * 255).astype(numpy.uint8)
    print(normalized_array.shape, type(normalized_array), normalized_array.dtype)
    print(normalized_array.min(), normalized_array.max(), normalized_array.mean(), numpy.median(normalized_array))

    print('Are arrays equal?', (numpy_image == normalized_array).all())


def integer_division():

    n = 35
    d = 3
    r = n - n%d

    print(r)


def visualize_tiff():

    # tiff_file = '/tmp/S1A_IW_GRDH_1SDV_20180704T163740_20180704T163805_022647_02742B_DF22.tif'
    # tiff_file = '/tmp/Flacksjon_study_area_sar_vh_2018-07-04_5.tif'
    tiff_file = '/tmp/bulk_export_flacksjon/S1A_IW_GRDH_1SDV_20180704T052317_20180704T052342_022640_0273F3_FD0A.tif'
    tiff_image = rio.open(tiff_file)
    print(tiff_image.read().shape)
    print(tiff_image.descriptions)
    # vv_index = tiff_image.descriptions.index('VV')
    vh_index = tiff_image.descriptions.index('VH')
    # print('VV index', vv_index)
    print('VH index', vh_index)
    file_name = os.path.basename(tiff_file)
    image_date = file_name[17:25]
    print(image_date)

    numpy_image = tiff_image.read([1])[0]
    show(numpy_image, title='')
    # print(numpy_image.shape, type(numpy_image))
    # # print(tiff_image)
    plt.show()


def count_color_pixels():
    im = Image.open('/Users/frape/KTH/Funding/MSCA-PF/2022/Figures/flacksjon_sar_mask.png')
    # im = Image.open('/Users/frape/KTH/Funding/MSCA-PF/2022/Figures/flacksjon_ndwi_mask.png')

    black = 0
    red = 0

    for pixel in im.getdata():
        if pixel == (0, 0, 0, 255):  # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
            black += 1
        else:
            red += 1
    print('black=' + str(black) + ', blue=' + str(red))


def transform_black_pixels_to_transparent():

    # Load image as Numpy array in BGR order
    na = cv2.imread('/Users/frape/KTH/Funding/MSCA-PF/2022/Figures/flacksjon_sar_mask.png')

    # Make a True/False mask of pixels whose BGR values sum to more than zero
    alpha = numpy.sum(na, axis=-1) > 0

    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = numpy.uint8(alpha * 255)

    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    res = numpy.dstack((na, alpha))

    # Save result
    cv2.imwrite('/tmp/transparent.png', res)


def main():
    # clip_matrix()
    # count_pixels_ratio()
    # generate_single_sample_csv()
    # transform_image_grayscale()
    # resize_image()
    # parallel_array_replace()
    # count_class_labels()
    # min_max_scale_nan()
    # normalize_tiff()
    # integer_division()
    # visualize_tiff()
    # count_color_pixels()
    transform_black_pixels_to_transparent()

start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
