import numpy as np
from scipy import ndimage


def fast_bilateral_filter(hdr_image, sigma_spatial=8.0, sigma_range=0.4, bins=100):
    """
    Applies fast bilateral filtering for tone mapping HDR images.

    Parameters:
    -----------
    hdr_image : numpy.ndarray
        The input HDR image (should be in linear RGB format).
    sigma_spatial : float
        Spatial standard deviation for the bilateral filter.
    sigma_range : float
        Range standard deviation for the bilateral filter.
    bins : int
        Number of bins for the intensity discretization.

    Returns:
    --------
    numpy.ndarray
        Tone mapped image in the [0, 1] range.
    """
    # Convert to luminance (using the Y component from XYZ)
    luminance = 0.2126 * hdr_image[:, :, 0] + 0.7152 * \
        hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]

    # Avoid log(0)
    epsilon = 1e-6
    log_luminance = np.log10(luminance + epsilon)

    # Determine range of log luminance values
    min_log_lum = log_luminance.min()
    max_log_lum = log_luminance.max()

    # Discretize log luminance into bins
    indices = np.clip(((log_luminance - min_log_lum) / (max_log_lum -
                      min_log_lum) * (bins - 1)).astype(int), 0, bins - 1)

    # Create bilateral grid
    grid = np.zeros((bins, hdr_image.shape[0], hdr_image.shape[1]))

    # Populate the grid
    for i in range(bins):
        grid[i] = (indices == i).astype(float)

    # Blur each slice of the grid using a Gaussian filter
    blurred_grid = ndimage.gaussian_filter(
        grid, sigma=[0, sigma_spatial, sigma_spatial])

    # Normalize the grid
    normalizer = np.maximum(blurred_grid, epsilon)
    normalized_grid = blurred_grid / normalizer

    # Compute output log luminance
    output_log_lum = np.zeros_like(log_luminance)
    for i in range(bins):
        bin_value = min_log_lum + (max_log_lum - min_log_lum) * i / (bins - 1)
        output_log_lum += bin_value * normalized_grid[i]

    # Apply range compression
    dynamic_range = max_log_lum - min_log_lum
    compression_factor = np.log10(1 + dynamic_range * sigma_range)

    compressed_log_lum = (output_log_lum - min_log_lum) / compression_factor

    # Convert back to linear domain
    output_lum = 10 ** compressed_log_lum

    # Scale the original color by the ratio of new luminance to original luminance
    ratio = output_lum / (luminance + epsilon)
    tone_mapped = np.zeros_like(hdr_image)
    for c in range(3):
        tone_mapped[:, :, c] = hdr_image[:, :, c] * ratio

    # Normalize to [0, 1] range
    tone_mapped = np.clip(tone_mapped, 0, 1)

    return tone_mapped


def tone_map(hdr_image, exposure=1.0, sigma_spatial=8.0, sigma_range=0.4, bins=100):
    """
    Tone map an HDR image using fast bilateral filtering.

    Parameters:
    -----------
    hdr_image : numpy.ndarray
        The input HDR image.
    exposure : float
        Exposure adjustment before tone mapping.
    sigma_spatial : float
        Spatial standard deviation for the bilateral filter.
    sigma_range : float
        Range standard deviation for the bilateral filter.
    bins : int
        Number of bins for the intensity discretization.

    Returns:
    --------
    numpy.ndarray
        Tone mapped image in the [0, 1] range.
    """
    # Apply exposure adjustment
    exposed_hdr = hdr_image * exposure

    # Apply fast bilateral filter for tone mapping
    tone_mapped = fast_bilateral_filter(
        exposed_hdr,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        bins=bins
    )

    return tone_mapped
