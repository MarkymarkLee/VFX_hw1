from utils.hdr_paul import generate_hdr_paul
from utils.image_alignment import align_images
from utils.tm_fast_bilateral_filtering import tone_map


def generate_hdr(method: str, images, exposure_times, num_samples=50, lambda_smooth=100):
    """Generate an HDR image using the specified method"""

    available_methods = ['paul']
    assert method in available_methods, f"{method} not in available methods: {available_methods}"

    if method == 'paul':
        return generate_hdr_paul(images, exposure_times, num_samples, lambda_smooth)
