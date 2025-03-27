import numpy as np
import cv2


def show_channel(name, channel):
    shown_channel = np.log(channel)
    shown_channel = (shown_channel - np.min(shown_channel)) / \
        (np.max(shown_channel) - np.min(shown_channel))
    shown_channel = shown_channel * 255
    shown_channel = np.clip(shown_channel, 0, 255).astype(np.uint8)
    shown_channel = cv2.resize(shown_channel, (672, 378))
    cv2.imshow(name, shown_channel)
    cv2.waitKey(0)


def gaussian_intensity(x, sigma_r):
    """Computes the Gaussian function for intensity differences."""
    # Ensure sigma_r is not zero to avoid division by zero
    sigma_r = max(sigma_r, 1e-6)
    return np.exp(-(x**2) / (sigma_r**2))

# Helper function for interpolation weights (Triangular/Hat function)


def interpolation_weight(img_intensity, i_j, segment_width):
    """Computes the interpolation weight using a triangular function."""
    # Ensure segment_width is not zero
    segment_width = max(segment_width, 1e-6)
    diff = np.abs(img_intensity - i_j)
    # Linear decay from 1 at center (i_j) to 0 at distance segment_width
    weight = np.maximum(0, 1 - diff / segment_width)
    return weight


def fast_bilateral_filter_channel(
    channel, sigma_s, sigma_r, z
):
    """
    Applies the Fast Bilateral Filter to a single image channel.

    Args:
        channel (np.ndarray): Input single-channel image (float32, range [0, inf)).
        sigma_s (float): Spatial standard deviation.
        sigma_r (float): Range (intensity) standard deviation.
        z (int): Downsampling factor.
        num_segments (int): Number of intensity segments.

    Returns:
        np.ndarray: Filtered single-channel image (float32).
    """
    h, w = channel.shape
    channel_orig = channel.copy()  # Keep original for interpolation weights

    min_val = np.min(channel)
    max_val = np.max(channel)
    val_range = max(max_val - min_val, 1e-6)  # Avoid zero range
    num_segments = int(val_range / sigma_r)
    num_segments = max(num_segments, 100)  # Ensure at least one segment
    segment_width = val_range / num_segments

    # --- Downsampling ---
    # Use INTER_AREA for downsampling to minimize aliasing
    channel_small = cv2.resize(
        channel,
        (w // z, h // z),
        interpolation=cv2.INTER_AREA,
    )

    # Note: The pseudocode mentions downsampling the kernel f_sigma_s.
    # Here, we apply a Gaussian filter with sigma_s directly on the
    # downsampled image, which is a common practice in implementations.
    # The effective spatial extent relative to the original image scales.

    # --- Initialize Output ---
    # J = 0 (as per pseudocode)
    J_final = np.zeros_like(channel_orig, dtype=np.float32)

    # --- Loop over Intensity Segments ---
    for j in range(num_segments):
        print(f"Processing segment {j+1}/{num_segments}", end='\r')
        # Calculate representative intensity for the segment
        # (Using segment center)
        i_j = min_val + j * segment_width

        # G'^j = g_sigma_r(I' - i^j)
        G_prime_j = gaussian_intensity(channel_small - i_j, sigma_r)

        # K'^j = G'^j (conv) f'_sigma_s/z
        # Apply spatial Gaussian filter. ksize=(0,0) derives size from sigma_s.
        # Add epsilon for numerical stability during division.
        K_prime_j = cv2.GaussianBlur(G_prime_j, (0, 0), sigma_s) + 1e-6

        # H'^j = G'^j * I'
        H_prime_j = G_prime_j * channel_small

        # H'*j = H'^j (conv) f'_sigma_s/z
        H_star_j = cv2.GaussianBlur(H_prime_j, (0, 0), sigma_s)

        # J'^j = H'*j / K'^j
        J_prime_j = H_star_j / K_prime_j

        # J^j = upsample(J'^j, z)
        # Use INTER_LINEAR for upsampling
        J_j = cv2.resize(
            J_prime_j, (w, h), interpolation=cv2.INTER_LINEAR
        )

        # J = J + J^j * InterpolationWeight(I, i^j)
        J_final += J_j * 1 / num_segments

    print()
    return J_final.astype(np.float32)


def fast_bilateral_tonemap(
    hdr_image,
    sigma_s=0.1,
    sigma_r=8,
    z=4,
    gamma=2.2,
):
    """
    Applies Fast Bilateral Filter based tone mapping to an HDR image.

    Args:
        hdr_image (np.ndarray): Input HDR image (float32, range [0, inf)).
        sigma_s (float): Spatial standard deviation for the filter.
        sigma_r (float): Range (intensity) standard deviation for the filter.
                           Adjust based on whether log domain is used.
                           (Smaller value like 0.4 is typical for log domain).
        z (int): Downsampling factor (e.g., 4 or 8).
        num_segments (int): Number of intensity segments (e.g., 8 or 16).
        gamma (float): Gamma correction factor for final display.

    Returns:
        np.ndarray: Tone-mapped LDR image (uint8, range [0, 255]).
    """
    if hdr_image.dtype != np.float32:
        hdr_image = hdr_image.astype(np.float32) / np.iinfo(
            hdr_image.dtype).max if hdr_image.dtype != np.float32 else hdr_image.astype(np.float32)

    if hdr_image.ndim == 3 and hdr_image.shape[2] == 3:
        # Process luminance only by converting to HSV
        # Convert RGB to HSV space
        hsv_image = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2HSV)

        # Extract channels
        h_channel, s_channel, v_channel = cv2.split(hsv_image)

        # Apply bilateral filter only to the Value (luminance) channel
        filtered_v_channel = fast_bilateral_filter_channel(
            v_channel, sigma_s, sigma_r, z
        )

        filtered_hsv = cv2.merge((h_channel, s_channel, filtered_v_channel))
        # Convert back to RGB
        filtered_image = cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2RGB)

    elif hdr_image.ndim == 2:
        print("Processing grayscale image...")
        filtered_image = fast_bilateral_filter_channel(
            hdr_image, sigma_s, sigma_r, z
        )
    else:
        raise ValueError(
            "Input image must be grayscale (H, W) or RGB (H, W, 3)")

    # --- Final Tone Mapping Adjustments ---
    # Normalize to [0, 1] range (simple approach)
    # More sophisticated global operators could be used here.
    # filtered_image = filtered_image / filtered_image.max() # Simple scaling - might clip highlights

    filtered_image = np.log(1 + filtered_image)  # Logarithmic scaling

    # Alternative: Apply gamma correction directly
    # Apply gamma correction for display
    # Clip negative values if any before power
    ldr_image = np.power(np.clip(filtered_image, 0, None), 1.0 / gamma)

    # Normalize to [0, 1] after gamma
    max_val = np.max(ldr_image)
    if max_val > 0:
        ldr_image = ldr_image / max_val

    # Convert to 8-bit LDR
    ldr_image_uint8 = (np.clip(ldr_image, 0, 1) * 255).astype(np.uint8)

    return ldr_image_uint8
