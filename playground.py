import os
import time

import cv2
import numpy
import pandas
import rasterio
import seaborn
import torch
from PIL import Image
import rasterio as rio
from dotenv import load_dotenv, dotenv_values
from matplotlib import pyplot as plt
from matplotlib.image import imread
from osgeo import gdal
from rasterio.plot import show
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from evaluation import semantic_segmentation_evaluator
from wetlands import viz_utils, jaccard_similarity


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


def filter_tiff_image():
    # Open the input raster file
    with rasterio.open(r"/tmp/Orebro lan_mosaic_2018-07-04_sar_VH.tif") as src:
        # Read the input raster values into a numpy array
        data = src.read(1)

        # Apply the median filter to the input data
        size = 9
        filtered_data = ndimage.median_filter(data, size=size)

        # Create a profile for the output raster
        profile = src.profile

    # Write the filtered data to a new output raster file
    with rasterio.open(f"/tmp/Orebro lan_mosaic_2018-07-04_sar_VH_filtered_{size}.tif", "w", **profile) as dst:
        dst.write(filtered_data, 1)


def visualize_ndwi_tiff():
    # tiff_file = '/tmp/20180127T102259_20180127T102314_T33VWG.tif'
    # tiff_file = '/tmp/20180320T101021_20180320T101021_T33VWG.tif'
    tiff_file = '/tmp/20180514T101029_20180514T101052_T33VWG.tif'
    # tiff_file = '/tmp/20180708T101031_20180708T101025_T33VWG.tif'
    # tiff_file = '/tmp/201810_rgb.tif'
    tiff_image = rio.open(tiff_file)

    print(tiff_image.read().shape)
    print(tiff_image.descriptions)

    # Read the grid values into numpy arrays
    red = tiff_image.read(4)
    green = tiff_image.read(3)
    blue = tiff_image.read(2)

    # Function to normalize the grid values
    def normalize(array):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min) / (array_max - array_min))

    red[red > 2000] = 2000
    green[green > 2000] = 2000
    blue[blue > 2000] = 2000

    # red[red > numpy.nanpercentile(red, 99)] = numpy.nanpercentile(tiff_image.read((4, 3, 2)), 99)
    # green[green > numpy.nanpercentile(green, 99)] = numpy.nanpercentile(tiff_image.read((4, 3, 2)), 99)
    # blue[blue > numpy.nanpercentile(blue, 99)] = numpy.nanpercentile(tiff_image.read((4, 3, 2)), 99)

    # Normalize the bands
    redn = normalize(red)
    greenn = normalize(green)
    bluen = normalize(blue)

    # Create RGB natural color composite
    rgb = numpy.dstack((redn, greenn, bluen))

    # Let's see how our color composite looks like
    plt.imshow(rgb)
    plt.show()


def detect_incomplete_image():

    base_dir = '/tmp/bulk_export_sar_karlhulteson/'
    image1 = 'S1A_IW_GRDH_1SDV_20160529T050703_20160529T050728_011469_011790_1826.tif'
    image2 = 'S1A_IW_GRDH_1SDV_20160529T050728_20160529T050753_011469_011790_A126.tif'
    image3 = 'S1A_IW_GRDH_1SDV_20151031T051530_20151031T051554_008392_00BDA5_0400.tif'

    path1 = base_dir + image1
    path2 = base_dir + image2
    path3 = base_dir + image3

    sar_polarization = os.getenv('SAR_POLARIZATION')
    # image_array = estimate_water.load_image(path3, sar_polarization)
    #
    # print(image_array.shape)


def check_for_nans_inside_images():
    masks_dir = os.getenv('NDWI_MASK_DIR')
    num_incomplete_images = 0
    for file in os.listdir(masks_dir):
        if file.endswith(".tif"):
            file_path = os.path.join(masks_dir, file)

            tiff_image = rio.open(file_path)
            # band = 'NDWI'

            numpy_image = tiff_image.read()

            # If the image is incomplete and has NaN values we ignore it
            if numpy.isnan(numpy_image).any():
                print(file_path)
                print('Incomplete image')
                num_incomplete_images += 1
                # return None

    print(f'There were a total of {num_incomplete_images} incomplete images')


def min_max_scaler():

    my_array = numpy.zeros((8,))
    print(my_array)
    config = dotenv_values(".env")
    # print(type(config))
    # print(config)

    import pprint
    import json
    pprint.pprint(config, sort_dicts=False)

    print(json.dumps(config, indent=4))


def transform_dataset():
    class experimental_dataset(Dataset):

        def __init__(self, data, transform):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data.shape[0])

        def __getitem__(self, idx):
            item = self.data[idx]
            item = self.transform(item)
            return item

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    x = torch.rand(8, 1, 2, 2)
    print(x)

    print('Begin of transformations')

    dataset = experimental_dataset(x, transform)

    for i, item in enumerate(dataset):
        print('Transformation ', i)
        print(item)


def merge_geotiff():

    # file1 = '/tmp/medium_tiff/medium_sweden_ndwi_mask-0000000000-0000000000.tif'
    # file2 = '/tmp/medium_tiff/medium_sweden_ndwi_mask-0000000000-0000023296.tif'
    folder = '/tmp/huge_sweden/'
    # files_to_mosaic = [file1, file2]  # However many you want.

    files_to_mosaic = [
        folder + 'huge_sweden_ndwi_mask-0000000000-0000000000.tif',
        folder + 'huge_sweden_ndwi_mask-0000000000-0000023296.tif',
        folder + 'huge_sweden_ndwi_mask-0000000000-0000046592.tif',
        folder + 'huge_sweden_ndwi_mask-0000023296-0000000000.tif',
        folder + 'huge_sweden_ndwi_mask-0000023296-0000023296.tif',
        folder + 'huge_sweden_ndwi_mask-0000023296-0000046592.tif',
        folder + 'huge_sweden_ndwi_mask-0000046592-0000000000.tif',
        folder + 'huge_sweden_ndwi_mask-0000046592-0000023296.tif',
        folder + 'huge_sweden_ndwi_mask-0000046592-0000046592.tif',
    ]

    g = gdal.Warp("/tmp/huge_sweden/all_huge_sweden_ndwi_mask.tif", files_to_mosaic, format="GTiff",
                  options=["COMPRESS=LZW", "TILED=YES"])  # if you want
    g = None  # Close file and flush to disk


def otsu_segmentation():

    # image_path = '/tmp/otsu/flacksjon_20180622.png'
    # image_path = '/tmp/otsu/flacksjon_20180424.png'
    # image_path = '/tmp/otsu/flacksjon_20200804.png'
    # image_path = '/tmp/otsu/flacksjon_20211121.png'
    image_path = '/tmp/bulk_export_sar_flacksjon/S1A_IW_GRDH_1SDV_20141005T052245_20141005T052310_002690_003012_53A0.tif'

    # # read the input image as a gray image
    # img = cv2.imread(image_path, 0)
    #
    # # Apply Otsu's thresholding
    # _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # display the output image
    # cv2.imshow("Otsu's Thresholding", th)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img = cv2.imread(image_path, 0)
    band = os.getenv('SAR_POLARIZATION')
    img = viz_utils.load_image(image_path, band)
    # min = img.min()
    # max = img.max()
    img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

    # Apply global (simple) thresholding on image
    # ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Apply Otsu's thresholding on image
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['Original Image', 'Global Thresholding (v=127)', "Otsu's Thresholding", 'Gaussian Filter + Otsu']
    # images = [img, th1, th2, th3]

    # for i in range(4):
    #     plt.subplot(2, 2, i + 1)
    #     plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.axis("off")
    plt.imshow(th2, 'gray')
    # plt.imshow(th3, 'gray')
    plt.show()


def calculate_iou():

    # file1 = '/tmp/tmp_otsu_selected/bw_20180505_annotated.png'
    # file1 = '/tmp/tmp_otsu_selected/20180505_pred_bw.png'
    date = '20180505'
    # date = '20180704'
    # date = '20211016'
    annotated_file = f'/tmp/tmp_otsu_selected/{date}_annotated_bw.png'
    im = Image.open(annotated_file).convert('L')
    patch_size = int(os.getenv('PATCH_SIZE'))
    new_width = (im.width // patch_size) * patch_size
    new_height = (im.height // patch_size) * patch_size
    im = im.crop((0, 0, new_width, new_height))
    # im = im.crop((0, 0, 896, 448))
    # im.load()
    annotated_data = numpy.array(im)

    # DeepAqua prediction
    prediction_file = f'/tmp/tmp_otsu_selected/{date}_pred_bw.png'
    img = Image.open(prediction_file).convert('L')
    img = img.crop((0, 0, new_width, new_height))
    prediction_data = numpy.array(img)
    print('\nDeepAqua')
    iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)

    # Otsu prediction
    prediction_file = f'/tmp/tmp_otsu_selected/{date}_otsu_bw.png'
    img = Image.open(prediction_file).convert('L')
    img = img.crop((0, 0, new_width, new_height))
    prediction_data = numpy.array(img)
    print('\nOtsu')
    iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)

    # Otsu Gaussian
    prediction_file = f'/tmp/tmp_otsu_selected/{date}_otsu_gaussian_bw.png'
    img = Image.open(prediction_file).convert('L')
    img = img.crop((0, 0, new_width, new_height))
    prediction_data = numpy.array(img)
    print('\nOtsu Gaussian')
    iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)

    # print('Hola')

    # image1 = imread(file1)
    # data = numpy.asarray(image1, dtype="int32")
    # print(image1)


def calculate_confusion_matrix(Y_pred, Y_val):
    FP = len(numpy.where(Y_pred - Y_val == 1)[0])
    FN = len(numpy.where(Y_pred - Y_val == -1)[0])
    TP = len(numpy.where(Y_pred + Y_val == 2)[0])
    TN = len(numpy.where(Y_pred + Y_val == 0)[0])
    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize=(6, 6))
    seaborn.heatmap(cmat / numpy.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()


def evaluate_semantic_segmentation_toy_example():

    pred_matrix = numpy.asarray([[1, 1], [1, 1]])
    gt_matrix = numpy.asarray([[1, 0], [1, 1]])

    calculate_confusion_matrix(pred_matrix, gt_matrix)
    confusion_matrix = semantic_segmentation_evaluator.calc_semantic_segmentation_confusion([pred_matrix], [gt_matrix])
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    print(TP, TN, FP, FN)

    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize=(6, 6))
    ax = seaborn.heatmap(cmat / numpy.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Negative'])
    plt.xlabel("Predictions")
    plt.ylabel("Real values")
    plt.show()


def count_files():
    # Group files by prefix before underscore and count them
    path = '/Users/frape/Projects/DeepWetlands/Datasets/wetlands/Annotated data/Students/All annotations'
    files = os.listdir(path)
    files.sort()
    # The prefix is the first part of the filename before the first underscore
    count_dict = {}
    for file in files:
        prefix = file.split('_')[0]
        if prefix in count_dict:
            count_dict[prefix] += 1
        else:
            count_dict[prefix] = 1
    print(count_dict)


def rename_annotated_files():
    # Rename files
    path = '/Users/frape/Projects/DeepWetlands/Datasets/wetlands/Annotated data/Students/All annotations'
    for filename in os.listdir(path):
        if filename.endswith('.tif'):
            new_filename = filename.replace('mask_sar_VH', 'annotated_vh')
            new_filename = new_filename.replace('iak_Hornborgasjon', 'Hornborgasjon_annotated_vh')
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))


def open_tiff():
    annotations_dir = '/Users/frape/Projects/DeepWetlands/Datasets/wetlands/Annotated data/Students/All annotations/'
    # tiff_file = os.path.join(annotations_dir, 'Svartadalen_annotated_vh_2014-10-05.tif')
    # tiff_file = os.path.join(annotations_dir, 'Hjalstaviken_annotated_vh_2014-10-12.tif')
    # tiff_file = os.path.join(annotations_dir, 'Hornborgasjon_annotated_vh_2015-06-02.tif')
    tiff_file = os.path.join(annotations_dir, 'Mossatrask_annotated_vh_2014-10-12.tif')
    tiff_image = rio.open(tiff_file)
    print(tiff_image.descriptions)

    tiff_file = os.path.join(annotations_dir, 'Hjalstaviken_annotated_vh_2014-10-12.tif')
    tiff_image = rio.open(tiff_file)
    print(tiff_image.descriptions)

    tiff_file = os.path.join(annotations_dir, 'Hornborgasjon_annotated_vh_2015-06-02.tif')
    tiff_image = rio.open(tiff_file)
    print(tiff_image.descriptions)

    tiff_file = os.path.join(annotations_dir, 'Mossatrask_annotated_vh_2014-10-12.tif')
    tiff_image = rio.open(tiff_file)
    print(tiff_image.descriptions)

    band = 'vis-gray'
    viz_utils.transform_ndwi_tiff_to_grayscale_png(annotations_dir, band)


def main():
    load_dotenv()

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
    # transform_black_pixels_to_transparent()
    # filter_tiff_image()
    # visualize_ndwi_tiff()
    # plot_image()
    # detect_incomplete_image()
    # check_for_nans_inside_images()
    # min_max_scaler()
    # transform_dataset()
    # merge_geotiff()
    # otsu_segmentation()
    # calculate_iou()
    # rename_annotated_files()
    # count_files()
    open_tiff()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
