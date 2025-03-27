import cv2
import numpy as np
from utils.hdr_paul import generate_hdr_paul
from utils.hdr_robertson import generate_hdr_robertson
from utils.hdr_nayar import generate_hdr_nayar
from utils.image_alignment import align_images
from utils.tm_fast_bilateral_filter import fast_bilateral_tonemap


def generate_hdr(method: str, images, exposure_times, num_samples=50, lambda_smooth=100):
    """Generate an HDR image using the specified method"""

    available_methods = ['paul', 'robertson', 'nayar']
    assert method in available_methods, f"{method} not in available HDR methods: {available_methods}"

    if method == 'paul':
        return generate_hdr_paul(images, exposure_times, num_samples, lambda_smooth)
    elif method == 'robertson':
        return generate_hdr_robertson(images, exposure_times)
    elif method == 'nayar':
        return generate_hdr_nayar(images)


def tonemap(hdr_image, method: str, gamma=2.2):
    """Tone map the HDR image using the specified method"""

    available_methods = ['fast_bilateral',
                         'reinhard', 'mantiuk', 'dragor']
    assert method in available_methods, f"{method} not in available Tonemap methods: {available_methods}"

    print(f"Tone mapping using {method} method...")

    image = None

    if method == 'fast_bilateral':
        return fast_bilateral_tonemap(hdr_image, gamma=gamma)
    elif method == 'reinhard':
        image = cv2.createTonemapReinhard(gamma=gamma).process(hdr_image)
    elif method == 'mantiuk':
        image = cv2.createTonemapMantiuk(gamma=gamma).process(hdr_image)
    elif method == 'dragor':
        image = cv2.createTonemapDrago(gamma=gamma).process(hdr_image)
    else:
        raise ValueError(f"Invalid tone mapping method: {method}")

    image = (image*255).astype(np.uint8)
    return image
