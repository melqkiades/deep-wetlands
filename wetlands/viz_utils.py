import numpy
import rasterio as rio
from PIL import Image
from matplotlib import pyplot as plt
from rasterio.plot import show
import geopandas as gpd


def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))


def visualize_sentinel2_image(geoboundary, shape_name, tif_file):

    # Open image file using Rasterio
    image = rio.open(tif_file)
    boundary = geoboundary[geoboundary.shapeName == shape_name]

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(15, 15))
    boundary.plot(facecolor="none", edgecolor='red', ax=ax)
    show(image, ax=ax)
    print(type(image))
    plt.show()


def visualize_tiles(geoboundary, shape_name, tif_file, tiles):
    print('Data dimensions: {}'.format(tiles.shape))
    tiles.head(3)

    image = rio.open(tif_file)
    fig, ax = plt.subplots(figsize=(15, 15))
    tiles.plot(facecolor="none", edgecolor='red', ax=ax)
    show(image, ax=ax)
    plt.show()

    image = rio.open(tif_file)
    boundary = geoboundary[geoboundary.shapeName == shape_name]

    # Geopandas sjoin function
    tiles = gpd.sjoin(tiles, boundary, op='within')

    fig, ax = plt.subplots(figsize=(15, 15))
    tiles.plot(facecolor="none", edgecolor='red', ax=ax)
    show(image, ax=ax)
    plt.show()


def visualize_image_from_file(tif_file):
    # Open image file using Rasterio
    image = rio.open(tif_file)

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(15, 15))
    show(image, ax=ax)
    print(image.count)
    print(image.read(1).shape)
    plt.show()


def show_crop(image, shape, title=''):
    """Crops an image based on the polygon shape.
    Reference: https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html#rasterio.mask.mask

    Args:
        image (str): Image file path (.tif)
        shape (geometry): The tile with which to crop the image
        title(str): Image title
    """

    with rio.open(image) as src:
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)
        # Crop out black (zero) border
        # _, x_nonzero, y_nonzero = np.nonzero(out_image)
        # out_image = out_image[
        #     :,
        #     np.min(x_nonzero):np.max(x_nonzero),
        #     np.min(y_nonzero):np.max(y_nonzero)
        # ]
        if out_image.shape[1] == 65:
            out_image = out_image[:, :-1, :]
        if out_image.shape[2] == 65:
            out_image = out_image[:, :, 1:]

        # Min-max scale the data to range [0, 1]
        # out_image[out_image > maxValue] = maxValue
        # out_image[out_image < minValue] = minValue
        # out_image = (out_image - minValue)/(maxValue - minValue)

        # Visualize image
        show(out_image, title=title)
        print(out_image.shape, type(out_image))
        print(out_image)
        plt.show()


def convert_ndwi_tiff_to_png(tiff_file, out_file):

    tiff_image = rio.open(tiff_file)
    image_array = tiff_image.read(tiff_image.descriptions.index('NDWI-mask') + 1)
    image_array[image_array > 0.5] = 1.00
    image_array[image_array <= 0.5] = 0.0

    plt.imshow(image_array, cmap='gray')
    img = Image.fromarray(numpy.uint8(image_array * 255), 'L')
    img.save(out_file)


def convert_rgb_tiff_to_png(tiff_file, out_file):

    tiff_image = rio.open(tiff_file)

    # Read the grid values into numpy arrays
    red_band = tiff_image.descriptions.index('B4') + 1
    green_band = tiff_image.descriptions.index('B3') + 1
    blue_band = tiff_image.descriptions.index('B2') + 1
    red = tiff_image.read(red_band)
    green = tiff_image.read(green_band)
    blue = tiff_image.read(blue_band)

    red[red > 2000] = 2000
    green[green > 2000] = 2000
    blue[blue > 2000] = 2000

    # Normalize the bands
    redn = normalize(red)
    greenn = normalize(green)
    bluen = normalize(blue)

    # Create RGB natural color composite
    rgb = numpy.dstack((redn, greenn, bluen))
    im = Image.fromarray((rgb * 255).astype("uint8"))
    im.save(out_file)
