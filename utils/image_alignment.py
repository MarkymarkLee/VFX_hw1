import numpy as np
import cv2


def create_median_threshold_bitmap(image, exclude_bits=0):
    """
    Create a median threshold bitmap from a grayscale image.

    Args:
        image: Grayscale image as numpy array
        exclude_bits: Number of least significant bits to exclude to reduce noise

    Returns:
        Binary threshold bitmap
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    # Exclude least significant bits to reduce noise
    if exclude_bits > 0:
        gray = gray >> exclude_bits << exclude_bits

    # Calculate median value
    median = np.median(gray)

    # Create binary image
    return gray >= median


def create_exclusion_bitmap(image, exclude_bits=1):
    """
    Create exclusion bitmap to ignore noisy pixels near the median value.

    Args:
        image: Grayscale image
        exclude_bits: Bits to exclude

    Returns:
        Exclusion mask where True indicates pixels to include
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    median = np.median(gray)
    threshold = 2 << (exclude_bits + 1)

    return np.abs(gray.astype(np.int32) - median) > threshold


def calculate_shift(tb1, tb2, eb1, eb2, max_shift=16):
    """
    Calculate shift between two threshold bitmaps.

    Args:
        tb1, tb2: Threshold bitmaps
        eb1, eb2: Exclusion bitmaps
        max_shift: Maximum allowed shift

    Returns:
        (x_shift, y_shift) tuple indicating best alignment
    """
    best_error = float('inf')
    best_shift = (0, 0)

    h, w = tb1.shape

    for y in range(-max_shift, max_shift + 1):
        for x in range(-max_shift, max_shift + 1):
            # Calculate boundaries for valid comparison area
            y1, y2 = max(0, y), min(h, h + y)
            x1, x2 = max(0, x), min(w, w + x)

            # Calculate corresponding boundaries for the shifted image
            y1s, y2s = max(0, -y), min(h, h - y)
            x1s, x2s = max(0, -x), min(w, w - x)

            # Extract valid regions for comparison
            region1_tb = tb1[y1:y2, x1:x2]
            region2_tb = tb2[y1s:y2s, x1s:x2s]

            region1_eb = eb1[y1:y2, x1:x2]
            region2_eb = eb2[y1s:y2s, x1s:x2s]

            # Combined exclusion mask
            combined_mask = region1_eb & region2_eb

            # Calculate error (pixels that don't match)
            error_map = (region1_tb != region2_tb) & combined_mask
            error = np.sum(error_map)

            if error < best_error:
                best_error = error
                best_shift = (x, y)

    return best_shift


def align_image_pyramid(reference, target, max_levels=6, max_shift=16):
    """
    Align images using image pyramid for coarse-to-fine alignment.

    Args:
        reference: Reference image
        target: Target image to align
        max_levels: Maximum pyramid levels
        max_shift: Maximum shift at the coarsest level

    Returns:
        Aligned target image
    """
    # Convert to 8-bit if needed
    reference_uint8 = reference.astype(
        np.uint8) if reference.dtype != np.uint8 else reference.copy()
    target_uint8 = target.astype(
        np.uint8) if target.dtype != np.uint8 else target.copy()

    # Initialize with no shift
    shift_x, shift_y = 0, 0

    # Create image pyramids
    ref_pyr = [reference_uint8]
    target_pyr = [target_uint8]

    # Build pyramid
    for i in range(max_levels - 1):
        ref_small = cv2.pyrDown(ref_pyr[-1])
        target_small = cv2.pyrDown(target_pyr[-1])

        if min(ref_small.shape[0], ref_small.shape[1]) < 16:
            break

        ref_pyr.append(ref_small)
        target_pyr.append(target_small)

    # Reverse to start from coarsest level
    ref_pyr.reverse()
    target_pyr.reverse()
    current_max_shift = max_shift

    # Process pyramid levels
    for level in range(len(ref_pyr)):
        # Create threshold and exclusion bitmaps
        ref_tb = create_median_threshold_bitmap(ref_pyr[level], exclude_bits=1)
        target_tb = create_median_threshold_bitmap(
            target_pyr[level], exclude_bits=1)

        ref_eb = create_exclusion_bitmap(ref_pyr[level], exclude_bits=1)
        target_eb = create_exclusion_bitmap(target_pyr[level], exclude_bits=1)

        # Calculate shift with search centered around current shift
        dx, dy = calculate_shift(
            ref_tb, target_tb, ref_eb, target_eb, max_shift=current_max_shift)

        # Update accumulated shift
        shift_x = shift_x * 2 + dx
        shift_y = shift_y * 2 + dy

        # Reduce max shift for finer levels
        current_max_shift = max(1, current_max_shift // 2)

    # Apply final shift to original image
    h, w = reference.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_image = cv2.warpAffine(target.astype(np.float32), M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    return aligned_image, (shift_x, shift_y)


def align_images(images):
    """
    Align a list of images using MTB algorithm.

    Args:
        images: List of images to align

    Returns:
        List of aligned images
    """
    if len(images) <= 1:
        return images

    # Use the middle image as reference
    reference_idx = len(images) // 2
    reference = images[reference_idx]

    aligned_images = []

    for i, img in enumerate(images):
        if i == reference_idx:
            aligned_images.append(img)  # Reference image stays the same
        else:
            aligned, shift = align_image_pyramid(reference, img)
            aligned_images.append(aligned)
            print(f"Image {i} aligned with shift {shift}")

    return aligned_images
