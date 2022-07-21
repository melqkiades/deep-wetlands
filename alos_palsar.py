import time

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_sar_image_asf_alos_palsar():

    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/ALOS-PALSAR/20090813/ALPSRP189311190-H2.2_UA/'
    # file_path = folder + 'HH-ALPSRP189311190-H2.2_UA.tif'
    file_path = folder + 'HV-ALPSRP189311190-H2.2_UA.tif'
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
    img = img.clip(vmin, vmax)
    # img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    # img_plt = plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    # plt.show()

    ax = plt.gca()
    img_plt = plt.imshow(img, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img_plt, cax=cax)

    # plt.savefig('/tmp/alos-palsar/alos-palsar-hh.png')
    plt.savefig('/tmp/alos-palsar/alos-palsar-hv.png')
    plt.cla()
    plt.clf()

    # Water mask
    # water_mask_threshold = 2000
    water_mask_threshold = 1000
    water_mask = img < water_mask_threshold
    water_ratio = water_mask.sum() / (img.shape[0] * img.shape[1])
    print('Water ratio', water_ratio)
    plt.imshow(water_mask)
    # plt.savefig('/tmp/alos-palsar/alos-palsar-hh_water_mask.png')
    plt.savefig('/tmp/alos-palsar/alos-palsar-hv_water_mask.png')
    # fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.cla()
    plt.clf()


def main():
    load_sar_image_asf_alos_palsar()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))