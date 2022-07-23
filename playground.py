import time

import numpy
import pandas
from PIL import Image


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


def main():
    # clip_matrix()
    # count_pixels_ratio()
    # generate_single_sample_csv()
    # transform_image_grayscale()
    resize_image()

start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
