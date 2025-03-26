import random
import numpy as np
from scipy.optimize import minimize


def sample_pixels(images, num_samples=50):
    """Sample pixels from images for response curve recovery without overlapping"""

    height, width, _ = images[0].shape

    # Create all possible pixel positions
    all_positions = [(y, x) for y in range(height) for x in range(width)]

    # Calculate max possible unique samples
    max_possible_samples = height * width
    num_samples = min(num_samples, max_possible_samples)

    # Use random.choices to sample without replacement
    pixels = random.sample(all_positions, num_samples)

    return pixels


def weight_function(z):
    """Weight function for pixel values
    Gives higher weight to values in the middle of the range
    """
    z_min, z_max = 0, 255
    if z <= (z_min + z_max) / 2:
        return z - z_min + 1
    return z_max - z + 1


def recover_response_curve(images, exposure_times, pixels, smoothness_lambda=100,
                           response_n=256, iterations=5):
    """
    Implement Mitsunaga and Nayar's Radiometric Self Calibration method

    Parameters:
    -----------
    images : list of numpy arrays
        List of differently exposed images
    exposure_times : list of float
        Exposure times for each image in seconds
    smoothness_lambda : float
        Smoothness regularization parameter
    response_n : int
        Number of points in the response curve (typically 256 for 8-bit images)
    iterations : int
        Number of iterations for optimization

    Returns:
    --------
    hdr_image : numpy array
        The reconstructed HDR image
    response_curve : numpy array
        The estimated camera response function
    """
    # Ensure images are numpy arrays
    images = [np.array(img) if not isinstance(
        img, np.ndarray) else img for img in images]

    # Get image dimensions and number of images
    num_images = len(images)
    height, width, channels = images[0].shape

    # Initialize response curve (start with a linear response)
    response_curve = np.zeros((channels, response_n))
    for i in range(response_n):
        response_curve[:, i] = np.power(
            i / (response_n - 1), 2.2)  # Initial gamma=2.2

    num_samples = len(pixels)
    # Extract sample pixels from all images
    sample_pixels = np.zeros(
        (num_images, num_samples, channels), dtype=np.uint8)
    for i in range(num_images):
        for j, (x, y) in enumerate(pixels):
            sample_pixels[i, j] = images[i][x, y]

    # For each color channel
    for c in range(channels):
        # Iterative optimization
        for iteration in range(iterations):
            # Prepare data for optimization
            Z = sample_pixels[:, :, c]  # Pixel values

            # Define the objective function for optimization
            def objective(response_params):
                # Reconstruct response curve from parameters
                # We use polynomial representation as in the paper
                g = np.zeros(response_n)
                for i in range(response_n):
                    x = i / (response_n - 1)
                    g[i] = np.sum([response_params[p] * (x ** p)
                                  for p in range(len(response_params))])

                # Normalize response curve
                g = g / g[-1]

                # Calculate error term
                error = 0
                for i in range(num_samples):
                    for j in range(num_images - 1):
                        for k in range(j + 1, num_images):
                            if Z[j, i] > 0 and Z[k, i] > 0:  # Avoid saturated pixels
                                error += (g[Z[j, i]] / exposure_times[j] -
                                          g[Z[k, i]] / exposure_times[k]) ** 2

                # Add smoothness constraint
                smoothness = 0
                for i in range(1, response_n - 1):
                    smoothness += (g[i-1] - 2*g[i] + g[i+1]) ** 2

                return error + smoothness_lambda * smoothness

            # Initial parameters (polynomial coefficients)
            initial_params = np.zeros(5)  # 4th degree polynomial
            initial_params[1] = 1.0  # Linear term

            # Optimize
            result = minimize(objective, initial_params, method='L-BFGS-B')

            # Update response curve for this channel
            for i in range(response_n):
                x = i / (response_n - 1)
                response_curve[c, i] = np.sum(
                    [result.x[p] * (x ** p) for p in range(len(result.x))])

            # Normalize
            response_curve[c, i] = response_curve[c, i] / response_curve[c, -1]

    return response_curve


def create_hdr_image(images, exposure_times, response_curves):
    """Create an HDR image using the recovered response curves"""
    height, width, channels = images[0].shape
    hdr_image = np.zeros(
        (height, width, channels), dtype=np.float32)

    # For normalization and weighting
    weight_sum = np.zeros(
        (height, width, channels), dtype=np.float32)

    for i, image in enumerate(images):
        delta_t = np.log(exposure_times[i])

        for c in range(3):
            # RGB
            response_curve = response_curves[c]

            # Convert pixel values to log radiance
            channel_data = image[:, :, c].astype(int)
            weights = np.array([weight_function(z) for z in range(256)])
            pixel_weights = weights[channel_data]

            # Get log radiance using the response curve
            log_radiance = response_curve[channel_data]

            # Adjusted for the exposure time
            log_radiance -= delta_t

            # Add weighted radiance to the HDR image
            hdr_image[:, :, c] += pixel_weights * log_radiance

            weight_sum[:, :, c] += pixel_weights

    # Get the weighted average
    eps = np.finfo(np.float32).eps  # To avoid division by zero
    hdr_image = hdr_image / (weight_sum + eps)

    # Convert from log domain to linear
    hdr_image = np.exp(hdr_image)

    return hdr_image


def generate_hdr_nayar(images, exposure_times, num_samples=50):
    """Generate an HDR image using Nayar's method"""
    print("Generating HDR image using Nayar's method...")
    max_samples = images[0].shape[0] * images[0].shape[1]
    num_samples = min(num_samples, max_samples)
    print("Sampling pixels")
    pixels = sample_pixels(images, num_samples)
    print(f"Recovering response curves with {num_samples} samples")
    response_curves = recover_response_curve(
        images, exposure_times, pixels)
    print("Creating HDR image")
    hdr_image = create_hdr_image(
        images, exposure_times, response_curves)
    return hdr_image, response_curves
