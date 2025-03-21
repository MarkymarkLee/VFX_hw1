import os
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from utils import generate_hdr, align_images, tone_map
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


def write_results(images, exposure_times, aligned_images, response_curves, hdr_image, tone_mapped):
    # 1. Plot response curves
    plt.figure(figsize=(10, 4))
    for i, channel in enumerate(['r', 'g', 'b']):
        plt.plot(range(256), response_curves[i], color=channel)
        plt.title('Response Curves')
        plt.xlabel('Pixel Value')
        plt.ylabel('Log Exposure')
    plt.tight_layout()
    plt.savefig('results/response_curve.jpg')
    plt.close()

    # 2. Plot original images
    num_images = len(images)
    columns = 5
    rows = num_images // columns + (1 if num_images % columns != 0 else 0)

    if num_images > columns:
        fig, axes = plt.subplots(
            rows, columns, figsize=(4 * columns, 3 * rows))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 3))
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
    plt.savefig('results/original_images.jpg')
    plt.close()

    # 3. Plot aligned images
    if num_images > columns:
        fig, axes = plt.subplots(
            rows, columns, figsize=(4 * columns, 3 * rows))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 3))
    for ax in axes:
        ax.axis('off')
    for i, img in enumerate(aligned_images):
        if num_images > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'Aligned - Exp: {exposure_times[i]:.4f}s')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/aligned_images.jpg')
    plt.close()

    # 4. Save HDR image
    cv2.imwrite('results/hdr_image.hdr',
                cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))

    # 5. Save tone-mapped result
    cv2.imwrite('results/result.jpg',
                cv2.cvtColor(tone_mapped, cv2.COLOR_RGB2BGR))


def main():
    data_folder = "data/memorial"

    # Read images and exposure times
    print("Reading images...")
    images, exposure_times = read_images(data_folder)

    num_samples = int(256 / (len(images) - 1)) + 1 + 5

    aligned_images = align_images(images)

    hdr_image, response_curves = generate_hdr(
        method='paul', images=images, exposure_times=exposure_times, num_samples=num_samples)

    # Tone map for display
    print("Tone mapping...")
    tonemap = cv2.createTonemap(10)
    ldrDurand = tonemap.process(hdr_image)
    # tone_mapped = ldrDurand
    tone_mapped = np.clip(ldrDurand * 255, 0, 255).astype('uint8')

    # tone_mapped = tone_map(hdr_image, sigma_spatial=8.0,
    #                        sigma_range=0.4, bins=100)

    # Save results
    print("Saving results...")
    write_results(images, exposure_times, aligned_images,
                  response_curves, hdr_image, tone_mapped)


if __name__ == "__main__":
    main()
