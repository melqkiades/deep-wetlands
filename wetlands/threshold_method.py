
import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from wetlands import viz_utils, noise_filters


def tiff_to_png():

    # tiff_file = '/tmp/bulk_export_ojesjon_sar/S1A_IW_GRDH_1SDV_20180704T052317_20180704T052342_022640_0273F3_FD0A.tif'
    # band = 'VH'

    tiff_file = '/tmp/bulk_export_ojesjon_ndwi_binary/20180704T103021_20180704T103023_T33VWG.tif'
    band = 'NDWI-collection'

    image = viz_utils.load_image(tiff_file, band, ignore_nan=True)
    if image is None:
        return
    plt.imsave(tiff_file + '_sar_bw.png', image, cmap='gray')


def dice_score(true_mask, pred_mask):
    """
    Calculate the Dice score between two binary masks.

    Parameters:
    true_mask (numpy array): Ground truth binary mask.
    pred_mask (numpy array): Predicted binary mask.

    Returns:
    float: Dice score between the two masks.
    """
    # Calculate intersection and union
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask)

    # Calculate Dice score
    if union == 0:
        return 1.0  # Both masks are empty
    else:
        return 2 * intersection / union


def dice_score_2(true_mask, pred_mask):
    """
    Calculate the Dice score between two binary masks.

    Parameters:
    true_mask (numpy array): Ground truth binary mask.
    pred_mask (numpy array): Predicted binary mask.

    Returns:
    float: Dice score between the two masks.
    """
    # Calculate intersection and union
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask)

    # Calculate Dice score
    if union == 0:
        return 1.0  # Both masks are empty
    else:
        return 2 * intersection / union


def dice_score_3(y_pred, y_true):
    lambda_var = 1.0

    intersection = (y_pred * y_true).sum()
    dice_loss = (2. * intersection + lambda_var) / (
            y_pred.sum() + y_true.sum() + lambda_var
    )

    return dice_loss


def pixel_accuracy(true_mask, pred_mask):
    """
    Calculate the pixel accuracy between two binary masks.

    Parameters:
    true_mask (numpy array): Ground truth binary mask.
    pred_mask (numpy array): Predicted binary mask.

    Returns:
    float: Pixel accuracy between the two masks.
    """
    # Calculate pixel accuracy
    return np.sum(true_mask == pred_mask) / true_mask.size


def find_optimal_threshold(sar_image, ground_truth_mask, thresholds):
    """
    Find the threshold that maximizes the Dice score for water detection in a SAR image.

    Parameters:
    sar_image (numpy array): Normalized SAR image.
    ground_truth_mask (numpy array): Ground truth binary mask for water.
    thresholds (list or numpy array): List of threshold values to evaluate.

    Returns:
    tuple: Optimal threshold and the corresponding Dice score.
    """
    optimal_threshold = None
    optimal_noise_filter = None
    max_dice_score = -1

    noise_filters_list = noise_filters.get_noise_filters()
    for noise_filter in noise_filters_list:
        sar_image_denoised = noise_filters_list[noise_filter](sar_image)
        for i, threshold in enumerate(thresholds):
            # Apply threshold to SAR image to create binary water prediction
            water_prediction = (sar_image_denoised < threshold).astype(int)
            plt.imsave(f'/tmp/water_masks/water_mask_bw_{i}.png', water_prediction, cmap='gray')

            # summary_statistics_sar = {
            #     "Min Value": np.min(water_prediction),
            #     "Max Value": np.max(water_prediction),
            #     "Mean Value": np.mean(water_prediction),
            #     "Median Value": np.median(water_prediction),
            #     "Standard Deviation": np.std(water_prediction)
            # }
            # print('Water prediction', summary_statistics_sar)

            # Calculate Dice score
            dice = dice_score(ground_truth_mask, water_prediction)
            # dice = dice_score_2(ground_truth_mask, water_prediction)
            # dice = dice_score_3(ground_truth_mask, water_prediction)
            # dice = pixel_accuracy(ground_truth_mask, water_prediction)

            # print(f'{i+1}/{len(thresholds)}: {threshold} -> {dice}')

            # Update optimal threshold if current Dice score is higher
            if dice > max_dice_score:
                optimal_threshold = threshold
                max_dice_score = dice
                optimal_noise_filter = noise_filter

    return optimal_threshold, max_dice_score, optimal_noise_filter


def plot_images(images, titles, cmap='gray', figsize=(12, 4)):
    """
    Plot a list of images with corresponding titles.

    Parameters:
    images (list): List of images to be plotted.
    titles (list): List of titles for each image.
    cmap (str): Colormap to be used for displaying the images.
    figsize (tuple): Size of the figure.
    """
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # Example usage with random binary masks
    # dice_score_example = dice_score(np.random.randint(0, 2, (5, 5)), np.random.randint(0, 2, (5, 5)))
    # print('Dice score example', dice_score_example)

    # Define a range of threshold values to evaluate
    thresholds = np.linspace(0, 1, 1000)

    sar_file = '/tmp/bulk_export_ojesjon_sar/S1A_IW_GRDH_1SDV_20180704T052317_20180704T052342_022640_0273F3_FD0A.tif'
    # sar_file = '/tmp/bulk_export_ojesjon_sar/S1A_IW_GRDH_1SDV_20200623T163752_20200623T163817_033147_03D707_6A78.tif'
    sar_band = 'VH'
    sar_image = viz_utils.load_image(sar_file, sar_band, ignore_nan=True)

    # Compute summary statistics for the SAR image
    summary_statistics_sar = {
        "Min Value": np.min(sar_image),
        "Max Value": np.max(sar_image),
        "Mean Value": np.mean(sar_image),
        "Median Value": np.median(sar_image),
        "Standard Deviation": np.std(sar_image)
    }
    print(summary_statistics_sar)

    # Plotting the distribution of values in the normalized SAR image
    plt.figure(figsize=(10, 5))
    sns.histplot(sar_image.flatten(), bins=100, kde=True)
    plt.title('Distribution of Values in Normalized SAR Image')
    plt.xlabel('Normalized Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    ndwi_file = '/tmp/bulk_export_ojesjon_ndwi_binary/20180704T103021_20180704T103023_T33VWG.tif'
    # ndwi_file = '/tmp/bulk_export_ojesjon_ndwi_binary/20200623T103031_20200623T103028_T33VWG.tif'
    ndwi_band = 'NDWI-collection'
    ndwi_image = viz_utils.load_image(ndwi_file, ndwi_band, ignore_nan=True)

    # Find the optimal threshold
    optimal_threshold, max_dice, optimal_noise_filter = find_optimal_threshold(sar_image, ndwi_image, thresholds)
    # print(optimal_threshold, max_dice, optimal_noise_filter)
    print('Optimal threshold:', optimal_threshold, '\tDice score:', max_dice, '\tNoise filter:', optimal_noise_filter)

    # Apply the optimal threshold to create the binary water prediction
    noise_filters_list = noise_filters.get_noise_filters()
    sar_image_denoised = noise_filters_list[optimal_noise_filter](sar_image)
    water_prediction_optimal = (sar_image_denoised < optimal_threshold).astype(int)

    # Plot the SAR image, ground truth, and water prediction
    plot_images(
        [sar_image, ndwi_image, water_prediction_optimal],
        ['SAR Image', 'Ground Truth (Water Mask)', 'Water Prediction (Optimal Threshold)'],
        cmap='gray'
    )

    # Plot SAR image
    plt.figure(figsize=(10, 10))
    plt.imshow(sar_image, cmap='gray')
    plt.title('SAR Image')
    plt.axis('off')
    plt.show()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
