
import numpy as np
import random


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
    """Weight function for pixel values following guassian distribution with range 0~1"""
    return np.exp(-4 * ((z - 127.5) / 127.5) ** 2)


def recover_response_curve(images, exposure_times, pixels, max_iterations=1000, threshold=1e-4):
    """Recover the camera response function using Robertson's method"""
    images = np.array(images, dtype=np.uint8)
    num_images = len(images)
    response_curves = []
    num_channels = images[0].shape[2]
    # Get the pixel values from the sampled locations
    z = np.zeros((len(images), len(pixels), num_channels), dtype=np.uint8)
    for i, img in enumerate(images):
        for j, (y, x) in enumerate(pixels):
            z[i, j] = img[y, x]

    w = np.array([weight_function(z) for z in range(256)])
    for c in range(num_channels):
        # Initialize g and E
        g = np.linspace(1, 256, 256, dtype=np.float32)
        g = g / g[128]  # Normalize so that g[128] = 1
        E = np.ones(len(pixels), dtype=np.float32)

        for i in range(max_iterations):

            prev_E = E.copy()
            # Update E
            exposure_times_array = np.array(exposure_times)
            w_z = np.array(w[z[:, :, c]])
            # Reshape exposure_times for proper broadcasting
            exposure_times_reshaped = exposure_times_array[:,
                                                           np.newaxis].astype(np.float32)
            numers = w_z * g[z[:, :, c]] * exposure_times_reshaped
            denoms = w_z * (exposure_times_reshaped ** 2)
            numer = np.sum(numers, axis=0)
            denom = np.sum(denoms, axis=0)
            E = numer / denom

            # Update g
            g = np.zeros(256, dtype=np.float32)
            for m in range(256):
                numer = 0
                denom = 0
                for j in range(num_images):
                    delta_t = exposure_times[j]
                    mask = z[j, :, c] == m
                    numers = mask * E * delta_t
                    denoms = np.array(mask)
                    numer += np.sum(numers, axis=0)
                    denom += np.sum(denoms, axis=0)
                if denom == 0:
                    g[m] = 0
                else:
                    g[m] = numer/denom

            g /= g[128] if g[128] != 0 else 1

            diff = np.mean(np.abs(E - prev_E))
            print(f"Iteration {i+1}, average change: {diff:.6f}", end='\r')

            if diff < threshold:
                print(f"\nConverged after {i+1} iterations")
                break

        g = np.log(g)
        response_curves.append(g)

    return response_curves


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


def generate_hdr_robertson(images, exposure_times, num_samples=100000):
    """Generate an HDR image using Robertson's method"""
    print("Generating HDR image using Robertson's method...")
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
