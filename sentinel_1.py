import time

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_settings_list():

    settings_list = []

    # Flip vertical
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190823/S1A_IW_GRDH_1SDV_20190823T162938_20190823T163003_028699_033FA1_9BCC.SAFE/measurement/'
    # # file_path = folder + 's1a-iw-grd-vv-20190823t162938-20190823t163003-028699-033fa1-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20190823t162938-20190823t163003-028699-033fa1-002.tiff'
    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190823/S1A_IW_GRDH_1SDV_20190823T162938_20190823T163003_028699_033FA1_9BCC.SAFE/measurement/'
    settings = {
        'folder': folder,
        'vv_file_path': folder + 's1a-iw-grd-vv-20190823t162938-20190823t163003-028699-033fa1-001.tiff',
        'vh_file_path': folder + 's1a-iw-grd-vh-20190823t162938-20190823t163003-028699-033fa1-002.tiff',
        'x_min': 6400,
        'x_max': 7100,
        'y_min': 9900,
        'y_max': 10500,
        'horizontal_flip': False,
        'vertical_flip': True,
        'type': 'vv',
    }
    settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20190823t162938'
    # settings_list.append(settings)

    # Flip horizontal
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190822/S1B_IW_GRDH_1SDV_20190822T052222_20190822T052247_017694_02149F_1F87.SAFE/measurement/'
    # file_path = folder + 's1b-iw-grd-vv-20190822t052222-20190822t052247-017694-02149f-001.tiff'
    # file_path = folder + 's1b-iw-grd-vh-20190822t052222-20190822t052247-017694-02149f-002.tiff'
    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190822/S1B_IW_GRDH_1SDV_20190822T052222_20190822T052247_017694_02149F_1F87.SAFE/measurement/'
    settings = {
        'folder': folder,
        'vv_file_path': folder + 's1b-iw-grd-vv-20190822t052222-20190822t052247-017694-02149f-001.tiff',
        'vh_file_path': folder + 's1b-iw-grd-vh-20190822t052222-20190822t052247-017694-02149f-002.tiff',
        'x_min': 22000,
        'x_max': 23000,
        'y_min': 15700,
        'y_max': 16500,
        'horizontal_flip': True,
        'vertical_flip': False,
        'type': 'vv',
    }
    settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20190822t052222'
    # settings_list.append(settings)

    # Flip horizontal
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T052325_20200319T052350_031740_03A919_CF30.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20200319t052325-20200319t052350-031740-03a919-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20200319t052325-20200319t052350-031740-03a919-002.tiff'
    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T052325_20200319T052350_031740_03A919_CF30.SAFE/measurement/'
    settings = {
        'folder': folder,
        'vv_file_path': folder + 's1a-iw-grd-vv-20200319t052325-20200319t052350-031740-03a919-001.tiff',
        'vh_file_path': folder + 's1a-iw-grd-vh-20200319t052325-20200319t052350-031740-03a919-002.tiff',
        'x_min': 22000,
        'x_max': 23000,
        'y_min': 600,
        'y_max': 1400,
        'horizontal_flip': True,
        'vertical_flip': False,
        'type': 'vv',
    }
    settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20200319t052325'
    settings_list.append(settings)

    # Flip vertical
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T163748_20200319T163813_031747_03A952_8BAF.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20200319t163748-20200319t163813-031747-03a952-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20200319t163748-20200319t163813-031747-03a952-002.tiff'
    folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T163748_20200319T163813_031747_03A952_8BAF.SAFE/measurement/'
    settings = {
        'folder': folder,
        'vv_file_path': folder + 's1a-iw-grd-vv-20200319t163748-20200319t163813-031747-03a952-001.tiff',
        'vh_file_path': folder + 's1a-iw-grd-vh-20200319t163748-20200319t163813-031747-03a952-002.tiff',
        'x_min': 17500,
        'x_max': 18500,
        'y_min': 9000,
        'y_max': 9800,
        'horizontal_flip': False,
        'vertical_flip': True,
        'type': 'vv',
    }
    settings['output_file'] = f's1a-iw-grd-vv-20200319t163748'
    # settings_list.append(settings)

    return settings_list


def plot_sentinel_1(settings):
    # Flip vertical
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190823/S1A_IW_GRDH_1SDV_20190823T162938_20190823T163003_028699_033FA1_9BCC.SAFE/measurement/'
    # # file_path = folder + 's1a-iw-grd-vv-20190823t162938-20190823t163003-028699-033fa1-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20190823t162938-20190823t163003-028699-033fa1-002.tiff'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190823/S1A_IW_GRDH_1SDV_20190823T162938_20190823T163003_028699_033FA1_9BCC.SAFE/measurement/'
    # settings = {
    #     'folder': folder,
    #     'vv_file_path': folder + 's1a-iw-grd-vv-20190823t162938-20190823t163003-028699-033fa1-001.tiff',
    #     'vh_file_path': folder + 's1a-iw-grd-vh-20190823t162938-20190823t163003-028699-033fa1-002.tiff',
    #     'x_min': 6400,
    #     'x_max': 7100,
    #     'y_min': 9900,
    #     'y_max': 10500,
    #     'horizontal_flip': False,
    #     'vertical_flip': True,
    #     'type': 'vv',
    # }
    # settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20190823t162938'

    # Flip horizontal
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190822/S1B_IW_GRDH_1SDV_20190822T052222_20190822T052247_017694_02149F_1F87.SAFE/measurement/'
    # file_path = folder + 's1b-iw-grd-vv-20190822t052222-20190822t052247-017694-02149f-001.tiff'
    # file_path = folder + 's1b-iw-grd-vh-20190822t052222-20190822t052247-017694-02149f-002.tiff'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20190822/S1B_IW_GRDH_1SDV_20190822T052222_20190822T052247_017694_02149F_1F87.SAFE/measurement/'
    # settings = {
    #     'folder': folder,
    #     'vv_file_path': folder + 's1b-iw-grd-vv-20190822t052222-20190822t052247-017694-02149f-001.tiff',
    #     'vh_file_path': folder + 's1b-iw-grd-vh-20190822t052222-20190822t052247-017694-02149f-002.tiff',
    #     'x_min': 22000,
    #     'x_max': 23000,
    #     'y_min': 15700,
    #     'y_max': 16500,
    #     'horizontal_flip': True,
    #     'vertical_flip': False,
    #     'type': 'vv',
    # }
    # settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20190822t052222'

    # Flip horizontal
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T052325_20200319T052350_031740_03A919_CF30.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20200319t052325-20200319t052350-031740-03a919-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20200319t052325-20200319t052350-031740-03a919-002.tiff'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T052325_20200319T052350_031740_03A919_CF30.SAFE/measurement/'
    # settings = {
    #     'folder': folder,
    #     'vv_file_path': folder + 's1a-iw-grd-vv-20200319t052325-20200319t052350-031740-03a919-001.tiff',
    #     'vh_file_path': folder + 's1a-iw-grd-vh-20200319t052325-20200319t052350-031740-03a919-002.tiff',
    #     'x_min': 22000,
    #     'x_max': 23000,
    #     'y_min': 600,
    #     'y_max': 1400,
    #     'horizontal_flip': True,
    #     'vertical_flip': False,
    #     'type': 'vv',
    # }
    # settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20200319t052325'

    # Flip vertical
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T163748_20200319T163813_031747_03A952_8BAF.SAFE/measurement/'
    # file_path = folder + 's1a-iw-grd-vv-20200319t163748-20200319t163813-031747-03a952-001.tiff'
    # file_path = folder + 's1a-iw-grd-vh-20200319t163748-20200319t163813-031747-03a952-002.tiff'
    # folder = '/Users/frape/Projects/DeepWetlands/Datasets/ASF/Sentinel-1/20200419/S1A_IW_GRDH_1SDV_20200319T163748_20200319T163813_031747_03A952_8BAF.SAFE/measurement/'
    # settings = {
    #     'folder': folder,
    #     'vv_file_path': folder + 's1a-iw-grd-vv-20200319t163748-20200319t163813-031747-03a952-001.tiff',
    #     'vh_file_path': folder + 's1a-iw-grd-vh-20200319t163748-20200319t163813-031747-03a952-002.tiff',
    #     'x_min': 17500,
    #     'x_max': 18500,
    #     'y_min': 9000,
    #     'y_max': 9800,
    #     'horizontal_flip': False,
    #     'vertical_flip': True,
    #     'type': 'vv',
    # }
    # settings['output_file'] = f's1a-iw-grd-{settings["type"]}-20200319t163748'

    #
    if settings['type'] == 'vv':
        settings['file_path'] = settings['vv_file_path']
        water_mask_threshold = 120
    if settings['type'] == 'vh':
        settings['file_path'] = settings['vh_file_path']
        water_mask_threshold = 60

    dataset = rasterio.open(settings['file_path'])

    print('Bounds:', dataset.bounds)
    print('Coordinate system:', dataset.crs)
    gcps, gcp_crs = dataset.gcps
    print(gcp_crs)
    img = dataset.read(1)
    img = img.astype(float)

    if settings['horizontal_flip']:
        img = np.fliplr(img)

    img[img == dataset.nodata] = np.nan  # Convert NoData to NaN
    # img = img[settings['y_min']:settings['y_max'], settings['x_min']:settings['x_max']]

    # Sentinel-1 coordinates
    # img = img[9900:10500, 6400:7100]
    # img = img[9000:9800, 17500:18500]
    # img = img[15700:16500, 22000:23000]
    # img = img[600:1400, 22000:23000]
    vmin, vmax = np.nanpercentile(img, (5, 95))  # 5-95% stretch
    print('Vmin, vmax', vmin, vmax)
    img = img.clip(vmin, vmax)
    ax = plt.gca()
    if settings['vertical_flip']:
        img_plt = plt.imshow(img, cmap='jet', origin='lower')
    else:
        img_plt = plt.imshow(img, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img_plt, cax=cax)
    # plt.show()

    output_file = '/tmp/sentinel-1/' + settings['output_file']
    output_file = output_file.replace('-vv-', f'-{settings["type"]}-')
    plt.show()
    # plt.savefig(output_file + '.png')
    # plt.cla()
    # plt.clf()

    # Water mask
    # water_mask = img < water_mask_threshold
    # water_ratio = water_mask.sum() / (img.shape[0] * img.shape[1])
    # print('Water ratio', water_ratio)
    # # print('Image shape', img.shape)
    # if settings['vertical_flip']:
    #     plt.imshow(water_mask,  origin='lower')
    # else:
    #     plt.imshow(water_mask)
    # plt.savefig(output_file + '_water_mask.png')
    # # fig.savefig(image_name, bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.cla()
    # plt.clf()


def main():
    settings_list = create_settings_list()

    # settings_list[0]['type'] = 'vh'
    # plot_sentinel_1(settings_list[0])
    for settings in settings_list:
        # settings['type'] = 'vv'
        # plot_sentinel_1(settings)
        settings['type'] = 'vh'
        plot_sentinel_1(settings)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))