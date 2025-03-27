import argparse
import os
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from utils import generate_hdr, align_images, tonemap
import json


def read_images(data_folder):
    """Read all images from data folder and their exposure times"""

    data = json.load(open(os.path.join(data_folder, 'exposure.json')))
    images = []
    exposure_times = []

    for file, exposure_time in data.items():
        path = os.path.join(data_folder, file)
        img = np.array(Image.open(path), dtype=np.float32)
        images.append(img)
        exposure_times.append(1/exposure_time)

    # Sort images and exposure times together based on exposure times
    sorted_indices = np.argsort(exposure_times)
    exposure_times = [exposure_times[i] for i in sorted_indices]
    images = [images[i] for i in sorted_indices]

    return images, np.array(exposure_times)


def write_results(result_dir, images, exposure_times, response_curves, hdr_image, tone_mapped):

    # Create results directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 1. Plot response curves
    plt.figure(figsize=(10, 4))
    for i, channel in enumerate(['r', 'g', 'b']):
        rc_len = len(response_curves[i])
        plt.plot(response_curves[i], range(256 - rc_len, 256), color=channel)
        plt.title('Response Curves')
        plt.ylabel('Pixel Value')
        plt.xlabel('log Radiance')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'response_curve.jpg'))
    plt.close()

    # 2. Plot original images
    num_images = len(images)
    columns = 3
    rows = num_images // columns + (1 if num_images % columns != 0 else 0)

    if num_images > columns:
        fig, axes = plt.subplots(
            rows, columns, figsize=(8 * columns, 6 * rows))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(8 * num_images, 6))
    for ax in axes:
        ax.axis('off')
    for i, img in enumerate(images):
        if num_images > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'Exposure Time: {exposure_times[i]:.4f}s')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'images.jpg'))
    plt.close()

    # 4. Save HDR image
    # Convert from RGB to BGR for OpenCV
    hdr_image_bgr = cv2.cvtColor(
        hdr_image.astype(np.float32), cv2.COLOR_RGB2BGR)
    # Save as proper HDR format (.hdr or .exr)
    cv2.imwrite(os.path.join(result_dir, 'hdr_image.hdr'), hdr_image_bgr)

    # 5. Save tone-mapped result
    cv2.imwrite(os.path.join(result_dir, 'result.jpg'),
                cv2.cvtColor(tone_mapped, cv2.COLOR_RGB2BGR))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate HDR image from a set of images')
    parser.add_argument('--data_folder', type=str, default='data/mine',
                        help='Folder containing images and exposure times')
    parser.add_argument('--hdr_method', type=str, default="paul",
                        help='Method to generate HDR image')
    parser.add_argument('--tonemap_method', type=str, default="fast_bilateral",
                        help='Method to tonemap HDR image')
    parser.add_argument('--align', type=bool, default=True,
                        help='Align images before HDR generation')
    parser.add_argument('--result_directory', type=str, default="results",
                        help='Directory to save results')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Gamma correction for display')
    return parser.parse_args()


def run_process(data_folder, hdr_method, tonemap_method, align, result_directory, gamma):
    # Read images and exposure times
    print("Reading images...")
    images, exposure_times = read_images(data_folder)

    if align:
        print("Aligning images...")
        images = align_images(images)

    # Generate HDR image
    hdr_image, response_curves = generate_hdr(
        method=hdr_method, images=images, exposure_times=exposure_times)

    if tonemap_method == 'all':
        all_tonemap_methods = ['fast_bilateral',
                               'reinhard', 'mantiuk', 'dragor']
        for method in all_tonemap_methods:
            tonemapped_image = tonemap(hdr_image, method=method, gamma=gamma)
            print("Saving results...")
            result_dir = os.path.join(result_directory, method)
            write_results(result_dir, images, exposure_times,
                          response_curves, hdr_image, tonemapped_image)
    else:
        tonemapped_image = tonemap(
            hdr_image, method=tonemap_method, gamma=gamma)
        print("Saving results...")
        write_results(result_directory, images, exposure_times,
                      response_curves, hdr_image, tonemapped_image)


def main():

    args = parse_args()

    data_folder = args.data_folder
    hdr_method = args.hdr_method
    tonemap_method = args.tonemap_method
    align = args.align
    result_directory = args.result_directory
    gamma = args.gamma

    run_process(data_folder, hdr_method, tonemap_method,
                align, result_directory, gamma)


if __name__ == "__main__":
    main()
