from utils.hdr_paul import generate_hdr_paul
from utils.hdr_robertson import generate_hdr_robertson
from utils.hdr_nayar import generate_hdr_nayar
from utils.image_alignment import align_images
from utils.tm_fast_bilateral_filtering import tone_map


def generate_hdr(method: str, images, exposure_times, num_samples=50, lambda_smooth=100):
    """Generate an HDR image using the specified method"""

    available_methods = ['paul', 'robertson', 'nayar']
    assert method in available_methods, f"{method} not in available methods: {available_methods}"

    if method == 'paul':
        return generate_hdr_paul(images, exposure_times, num_samples, lambda_smooth)
    elif method == 'robertson':
        return generate_hdr_robertson(images, exposure_times)
    elif method == 'nayar':
        return generate_hdr_nayar(images, exposure_times)
