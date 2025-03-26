import numpy as np
import cv2


def align_MTB(images, max_shift=64, noise_threshold=4):
    """
    Aligns a list of images using the Median Threshold Bitmap (MTB) algorithm.

    Parameters:
    -----------
    images : list of numpy.ndarray
        List of input images (grayscale or color). If color, they will be converted to grayscale.
    max_shift : int
        Maximum allowed shift in pixels (should be power of 2).
    noise_threshold : int
        Threshold value for exclusion bitmap to handle noise (pixels within +/- this value of the median).

    Returns:
    --------
    list of tuples
        List of (x, y) integer offsets for each image relative to the reference image (first image).
    """
    if not images:
        return []

    N = len(images)

    # Convert all images to grayscale if necessary
    gray_images = []
    for img in images:
        if len(img.shape) == 3:
            # Convert to grayscale using the formula from the paper
            gray = np.round(
                (54 * img[:, :, 2] + 183 * img[:, :, 1] + 19 * img[:, :, 0]) / 256).astype(np.uint8)
            gray_images.append(gray)
        else:
            gray_images.append(img)

    # Create image pyramids
    pyramids = []
    for gray in gray_images:
        pyramid = [gray]
        shift_bits = int(np.log2(max_shift))
        for i in range(shift_bits):
            pyramid.append(cv2.resize(
                pyramid[-1], (pyramid[-1].shape[1] // 2, pyramid[-1].shape[0] // 2)))
        pyramid.reverse()  # Smallest to largest
        pyramids.append(pyramid)

    # Define reference image (first one)
    ref_pyramid = pyramids[N // 2]

    # Calculate offsets for each image relative to the reference
    offsets = [None] * N
    offsets[N // 2] = (0, 0)

    for i in range(0, len(images)):
        if i == N // 2:
            continue

        target_pyramid = pyramids[i]
        cur_offset = [0, 0]

        # Process each level of the pyramid
        for level in range(len(ref_pyramid)):
            ref_img = ref_pyramid[level]
            target_img = target_pyramid[level]

            # Double the offset from the previous level
            if level > 0:
                cur_offset[0] *= 2
                cur_offset[1] *= 2

            # Calculate median values
            ref_median = np.median(ref_img)
            target_median = np.median(target_img)

            # Create threshold bitmaps
            ref_tb = (ref_img > ref_median).astype(np.uint8)
            target_tb = (target_img > target_median).astype(np.uint8)

            # Create exclusion bitmaps
            ref_eb = (np.abs(ref_img.astype(np.int16) - ref_median)
                      > noise_threshold).astype(np.uint8)
            target_eb = (np.abs(target_img.astype(np.int16) -
                         target_median) > noise_threshold).astype(np.uint8)

            # Find the best offset at this level
            min_err = float('inf')
            best_dx, best_dy = 0, 0

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    xs = cur_offset[0] + dx
                    ys = cur_offset[1] + dy

                    # Shift the target bitmaps
                    M = np.float32([[1, 0, xs], [0, 1, ys]])
                    shifted_tb = cv2.warpAffine(
                        target_tb, M, (target_tb.shape[1], target_tb.shape[0]))
                    shifted_eb = cv2.warpAffine(
                        target_eb, M, (target_eb.shape[1], target_eb.shape[0]))

                    # Calculate difference between bitmaps
                    diff_b = cv2.bitwise_xor(ref_tb, shifted_tb)

                    # Apply exclusion bitmaps
                    diff_b = cv2.bitwise_and(diff_b, ref_eb)
                    diff_b = cv2.bitwise_and(diff_b, shifted_eb)

                    # Calculate error
                    err = np.sum(diff_b)

                    if err < min_err:
                        min_err = err
                        best_dx, best_dy = xs, ys

            # Update current offset
            cur_offset[0] = best_dx
            cur_offset[1] = best_dy

        offsets[i] = (cur_offset[0], cur_offset[1])
        print(f"Image {i} offset: {offsets[i]}")

    return offsets


def align_images(images):
    """
    Apply calculated offsets to align images.

    Parameters:
    -----------
    images : list of numpy.ndarray
        List of input images.
    offsets : list of tuples
        List of (x, y) offsets for each image.

    Returns:
    --------
    list of numpy.ndarray
        Aligned images.
    """
    aligned_images = []
    offsets = align_MTB(images)

    for i, img in enumerate(images):
        offset_x, offset_y = offsets[i]

        # Apply transformation matrix
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        aligned_images.append(aligned)

    return aligned_images
