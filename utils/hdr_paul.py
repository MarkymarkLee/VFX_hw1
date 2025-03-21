
import numpy as np


def sample_pixels(images, num_samples=50):
    """Sample pixels from images for response curve recovery"""
    height, width, channels = images[0].shape
    pixels = []
    for i in range(num_samples):
        y = np.random.randint(height)
        x = np.random.randint(width)
        pixels.append((y, x))
    return pixels


def weight_function(z):
    """Weight function for pixel values
    Gives higher weight to values in the middle of the range
    """
    z_min, z_max = 0, 255
    if z <= (z_min + z_max) / 2:
        return z - z_min + 1
    return z_max - z + 1


def recover_response_curve(images, exposure_times, pixels, lambda_smooth=100):
    """Recover the camera response function using Debevec's method"""
    num_images = len(images)
    num_pixels = len(pixels)
    z_max = 255

    # Setup the system of equations
    n_unknowns = z_max + 1 + num_pixels  # g(0)...g(255) + ln(E) for each pixel

    # Count equations
    n_equations = num_pixels * num_images + z_max

    # For each color channel
    response_curves = []

    for channel in range(3):  # RGB
        k = 0
        A = np.zeros((n_equations, n_unknowns))
        b = np.zeros(n_equations)

        # Data fitting equations
        for i, pixel in enumerate(pixels):
            y, x = pixel
            for j, image in enumerate(images):
                z = int(image[y, x, channel])
                w = float(weight_function(z+1))
                delta_t = np.log(exposure_times[j])

                # ln(Ei) = g(Zij) - ln(Δtj)
                A[k, z] = w
                A[k, z_max + i] = -w
                b[k] = w * delta_t
                k += 1

        # Smoothness equations with regularization term
        for i in range(z_max-2):
            w = weight_function(i)
            A[k, i] = lambda_smooth * w
            A[k, i + 1] = -2 * lambda_smooth * w
            A[k, i + 2] = lambda_smooth * w
            k += 1

        # Fix the curve by setting its middle value to 0
        A[k, 128] = 0
        k += 1

        # Solve the system using least squares
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        # Extract the response curve (first z_max+1 elements)
        response_curve = x[:z_max+1]
        response_curves.append(response_curve)

    return response_curves


def create_hdr_image(images, exposure_times, response_curves):
    """Create an HDR image using the recovered response curves"""
    height, width, channels = images[0].shape
    hdr_image = np.zeros((height, width, channels), dtype=np.float32)

    # For normalization and weighting
    weight_sum = np.zeros((height, width, channels), dtype=np.float32)

    for i, image in enumerate(images):
        delta_t = np.log(exposure_times[i])

        for c in range(3):  # RGB
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


def generate_hdr_paul(images, exposure_times, num_samples=50, lambda_smooth=100):
    """Generate an HDR image using Debevec's method"""
    print("Generating HDR image using Debevec's method...")
    print(f"Number of samples: {num_samples}, Lambda smooth: {lambda_smooth}")
    print("Sampling pixels")
    pixels = sample_pixels(images, num_samples)
    print("Recovering response curves")
    response_curves = recover_response_curve(
        images, exposure_times, pixels, lambda_smooth)
    print("Creating HDR image")
    hdr_image = create_hdr_image(images, exposure_times, response_curves)
    return hdr_image, response_curves
