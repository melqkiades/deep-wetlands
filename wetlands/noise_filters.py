import numpy as np
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter, variance, generic_filter
from skimage.restoration import estimate_sigma, denoise_nl_means, denoise_tv_chambolle


def apply_mean_filter(image, size=3):
    """
    Apply a mean filter to an image.

    Parameters:
    image (numpy array): The input image.
    size (int): The size of the filtering kernel.

    Returns:
    numpy array: The filtered image.
    """
    return uniform_filter(image, size=size)


def apply_median_filter(image, size=3):
    """
    Apply a median filter to an image.

    Parameters:
    image (numpy array): The input image.
    size (int): The size of the filtering kernel.

    Returns:
    numpy array: The filtered image.
    """
    return median_filter(image, size=size)


def apply_gaussian_filter(image, sigma=1):
    """
    Apply a Gaussian filter to an image.

    Parameters:
    image (numpy array): The input image.
    sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    numpy array: The filtered image.
    """
    return gaussian_filter(image, sigma=sigma)


def apply_lee_filter(image, size=3):
    """
    Apply a Lee filter to a SAR image for speckle noise reduction.

    Parameters:
    image (numpy array): The input SAR image.
    size (int): The size of the filtering kernel.

    Returns:
    numpy array: The filtered image.
    """
    img_mean = uniform_filter(image, size)
    img_sqr_mean = uniform_filter(image ** 2, size)
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(image)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (image - img_mean)

    return img_output


def frost_filter(image, size=3, damping_factor=2):
    """
    Apply a Frost filter to a SAR image for speckle noise reduction.

    Parameters:
    image (numpy array): The input SAR image.
    size (int): The size of the filtering kernel.
    damping_factor (float): Damping factor for the filter.

    Returns:
    numpy array: The filtered image.
    """
    def frost_kernel(data, damping_factor):
        center = data[len(data) // 2]
        diff = data - center
        weights = np.exp(-(diff ** 2) / damping_factor)
        result = np.sum(data * weights) / np.sum(weights)
        return result

    return generic_filter(image, function=frost_kernel, size=size, extra_arguments=(damping_factor,))


def apply_kuan_filter(image, size=3):
    """
    Apply a Kuan filter to a SAR image for speckle noise reduction.

    Parameters:
    image (numpy array): The input SAR image.
    size (int): The size of the filtering kernel.

    Returns:
    numpy array: The filtered image.
    """
    img_mean = uniform_filter(image, size)
    img_sqr_mean = uniform_filter(image ** 2, size)
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(image)

    kuan_filter = (img_variance - overall_variance) / (img_variance + overall_variance)
    img_output = img_mean + kuan_filter * (image - img_mean)

    return img_output


def apply_non_local_means_filter(image, patch_size=7, patch_distance=11, h=0.1):
    """
    Apply a Non-Local Means filter to an image.

    Parameters:
    image (numpy array): The input image.
    patch_size (int): Size of patches used for denoising.
    patch_distance (int): Maximal distance in pixels where to search patches used for denoising.
    h (float): Cut-off distance (in gray levels). The higher h, the more permissive one is in accepting patches.

    Returns:
    numpy array: The filtered image.
    """
    sigma_est = np.mean(estimate_sigma(image, multichannel=False))
    denoised_img = denoise_nl_means(image, h=h*sigma_est, fast_mode=True,
                                    patch_size=patch_size, patch_distance=patch_distance, multichannel=False)
    return denoised_img


def apply_total_variation_denoising(image, weight=0.1, max_iter=200):
    """
    Apply total variation denoising to an image.

    Parameters:
    image (numpy array): The input image.
    weight (float): Denoising weight. The greater weight, the more denoising (at the expense of fidelity to input).
    max_iter (int): Maximal number of iterations used for optimization.

    Returns:
    numpy array: The denoised image.
    """
    denoised_img = denoise_tv_chambolle(image, weight=weight, max_num_iter=max_iter, multichannel=False)
    return denoised_img

def get_noise_filters():
    noise_filters = {
        'mean': apply_mean_filter,
        'median': apply_median_filter,
        'gaussian': apply_gaussian_filter,
        # 'lee': apply_lee_filter,
        # 'frost': frost_filter,
        # 'kuan': apply_kuan_filter,
        # 'non_local_means': apply_non_local_means_filter,
        # 'total_variation': apply_total_variation_denoising,
    }

    return noise_filters



