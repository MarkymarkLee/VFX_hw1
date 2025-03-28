import random
import numpy as np


def sample_pixels(images, num_samples=50):
    """Sample pixels from images for response curve recovery without overlapping"""

    height, width, channel = images[0].shape

    # sample pixels only from points that are increasing in value
    total_mask = np.ones((height, width), dtype=np.bool)
    for i in range(len(images) - 1):
        mask = images[i] < images[i+1]
        total_mask = np.logical_and(total_mask, np.all(mask, axis=2))

    for i in range(len(images)):
        mask = images[i] != 0
        total_mask = np.logical_and(total_mask, np.all(mask, axis=2))

    # Create all possible pixel positions
    all_positions = [(y, x) for y in range(height)
                     for x in range(width) if total_mask[y, x]]
    print(f"Total possible samples: {len(all_positions)}")

    # Calculate max possible unique samples
    max_possible_samples = len(all_positions)
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


def recover_response_curve(images, pixels, max_iterations=500, err_threshold=1e-2):
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
    response_curve : numpy array
        The estimated camera response function
    """
    response_n = 256
    # Ensure images are numpy arrays
    images = [np.array(img) if not isinstance(
        img, np.ndarray) else img for img in images]

    # Get image dimensions and number of images
    num_images = len(images)
    height, width, channels = images[0].shape

    response_curve = np.zeros((channels, response_n))

    num_samples = len(pixels)
    # Extract sample pixels from all images
    sample_pixels = np.zeros(
        (num_images, num_samples, channels), dtype=np.uint8)
    for i in range(num_images):
        for j, (x, y) in enumerate(pixels):
            sample_pixels[i, j] = images[i][x, y]

    best_degree = 0
    best_error = float('inf')
    best_response_curve = None
    best_exposures = None
    # For each degree of response curve
    for degree in range(3, 7):
        print(f"Optimizing for degree {degree} polynomial")
        total_error = 0
        # For each color channel
        for c in range(channels):

            Z = sample_pixels[:, :, c]  # Pixel values
            R = np.ones(num_images-1).astype(np.float32) / 2
            prev_g = np.zeros(256)
            cur_error = 0

            # Iterative optimization
            for iteration in range(max_iterations):

                # create linear system
                A = np.zeros((degree+1, degree+1))
                B = np.zeros(degree+1)

                table = np.zeros((num_images-1, num_samples, degree))
                for q in range(num_images - 1):
                    for p in range(num_samples):
                        for d in range(degree):
                            m_pq = Z[q, p] / 255
                            m_pq1 = Z[q+1, p] / 255
                            table[q, p, d] = m_pq ** d - R[q] * m_pq1 ** d

                for d in range(degree):
                    A[d, -1] = -1
                    # add d_epsilon/d_d = 0 to the system
                    for q in range(num_images - 1):
                        for p in range(num_samples):
                            for dd in range(degree):
                                A[d, dd] += table[q, p, d] * table[q, p, dd]

                # add sum(x) = 1 to the system
                A[degree, :-1] = 1
                B[degree] = 1

                # Solve the system
                c_n = np.linalg.solve(A, B)[:-1]

                g = np.zeros(response_n)
                for i in range(response_n):
                    x = i / 255
                    g[i] = np.sum([c_n[p] * (x ** p)
                                   for p in range(len(c_n))])

                # g = (g - np.min(g) + 0.01) / (np.max(g) - np.min(g))

                cur_error = 0
                # Update R with new response curve
                for j in range(num_images - 1):
                    p_j = g[Z[j, :]]
                    p_j1 = g[Z[j+1, :]]
                    cur_error += np.sum(np.power(p_j - R[j] * p_j1, 2))
                    mask = p_j1 != 0
                    p_j1 = p_j1[mask]
                    p_j = p_j[mask]
                    R[j] = np.sum(p_j/p_j1) / num_samples

                # Update initial parameters for next iteration
                # initial_params = result.x
                error = np.abs(g - prev_g)
                error = error > err_threshold
                error = np.sum(error)
                prev_g = g

                print(
                    f"Iteration {iteration}, Diff: {error}, error={cur_error}, R={R}", end='\r')

                if error == 0:
                    response_curve[c] = g
                    exposures = np.ones((channels, num_images))
                    exposures[0] = 1
                    for i in range(1, num_images):
                        exposures[c, i] = R[i-1] / exposures[c, i-1]
                    # exposures = exposures / np.sum(exposures)
                    print(f"Channel {c}, R={R}" + " " * 50)

                    break

            total_error += cur_error

        if total_error < best_error:
            best_error = total_error
            best_degree = degree
            best_response_curve = response_curve
            best_exposures = exposures
            print(f"New best degree: {best_degree}")

        print(f"Total error for degree {degree}: {total_error}")

    print(f"Best degree: {best_degree}, Best error: {best_error}")

    return best_response_curve, best_exposures


def create_hdr_image(images, exposures, response_curves):
    """Create an HDR image using the recovered response curves"""
    height, width, channels = images[0].shape
    hdr_image = np.zeros(
        (height, width, channels), dtype=np.float32)

    # For normalization and weighting
    weight_sum = np.zeros(
        (height, width, channels), dtype=np.float32)

    M = np.array(images).astype(np.uint8)

    M_r = M[:, :, :, 0].flatten()
    M_g = M[:, :, :, 1].flatten()
    M_b = M[:, :, :, 2].flatten()

    I_r = response_curves[0][M_r]
    I_g = response_curves[1][M_g]
    I_b = response_curves[2][M_b]

    # Calculate sums for robust ratio estimation
    sum_MrIg = np.sum(M_r * I_g)
    sum_MgIr = np.sum(M_g * I_r)
    sum_MbIg = np.sum(M_b * I_g)
    sum_MgIb = np.sum(M_g * I_b)

    eps = 1e-8

    # Calculate relative scales s_r = kr/kg, s_b = kb/kg
    s_r = sum_MrIg / (sum_MgIr + eps)
    s_b = sum_MbIg / (sum_MgIb + eps)

    # Return scales relative to Green (G scale = 1.0)
    scales = np.array([s_r, 1.0, s_b], dtype=np.float32)
    print(f"Calculated color scales (R, G, B relative to G): {scales}")
    color_scale = scales / np.sum(scales)

    for i, image in enumerate(images):

        for c in range(3):
            # RGB
            response_curve = response_curves[c]

            # Convert pixel values to log radiance
            channel_data = image[:, :, c].astype(int)
            weights = np.array([weight_function(z)
                               for z in range(256)])
            pixel_weights = weights[channel_data]

            radiance = response_curve[channel_data]

            # Add weighted radiance to the HDR image
            hdr_image[:, :, c] += pixel_weights * \
                radiance / exposures[c][i] * color_scale[c]
            weight_sum[:, :, c] += pixel_weights * color_scale[c]

    # Get the weighted average
    eps = np.finfo(np.float32).eps  # To avoid division by zero
    hdr_image = hdr_image / (weight_sum + eps)

    return hdr_image


def generate_hdr_nayar(images, num_samples=1000):
    """Generate an HDR image using Nayar's method"""
    print("Generating HDR image using Nayar's method...")
    max_samples = images[0].shape[0] * images[0].shape[1]
    num_samples = min(num_samples, max_samples)

    print("Sampling pixels")
    pixels = sample_pixels(images, num_samples)
    print(f"Recovering response curves with {len(pixels)} samples")
    response_curves, exposures = recover_response_curve(
        images, pixels)
    print("Creating HDR image")
    hdr_image = create_hdr_image(
        images, exposures, response_curves)
    return hdr_image, response_curves
