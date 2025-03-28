# High Dynamic Range Imaging (HDRI)

This repository contains an implementation of high dynamic range (HDR) imaging algorithms that combine multiple exposures of the same scene to create a wider dynamic range image.

## Overview

The project implements several HDR generation algorithms, image alignment techniques, and tone mapping methods to create high-quality HDR images from a series of differently exposed photographs.

## Features

-   **HDR Generation Methods**:

    -   Debevec & Malik's method (implemented as 'paul')
    -   Robertson's method
    -   Mitsunaga & Nayar's Radiometric Self Calibration method

-   **Image Alignment**:

    -   Median Threshold Bitmap (MTB) alignment for robust registration of differently exposed images

-   **Tone Mapping Operators**:
    -   Fast Bilateral Filter technique
    -   Standard OpenCV operators (Reinhard, Durand, Mantiuk, Drago)

## Dataset Structure

The repository supports multiple datasets with exposure information stored in JSON format:

-   `data/mine/` - Personal dataset with 6 exposures
-   `data/memorial/` - Memorial dataset with 16 exposures
-   `data/palace/` - Palace dataset with 4 exposures
-   `data/sofa/` - Sofa dataset with 12 exposures

If you want to use your own image, the folder structure should be

```
--data
    --exposure.json
    --all the original images
```

exposure.json should include all images and be structured as

```
{
    "image_file_name": 0.002 # shutter speed for this image
}
```

## Usage

### Environment setup

Required packages are listed in `requirements.txt`:

Install with:

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python main.py --data_folder data/mine \
               --hdr_method paul \
               --tonemap_method fast_bilateral \
               --align True \
               --result_directory "results" \
               --gamma 2.2
```

### Recreate results folder

Running run.py creates the results folder in the repo

```bash
python run.py
```

## Algorithm Details

### MTB Image Alignment

Aligns images by comparing median threshold bitmaps, which are robust to exposure changes, using a pyramid-based approach for efficiency.

### Debevec & Malik Method

Recovers camera response functions using a system of equations with smoothness constraints, then uses these functions to compute radiance values.

### Robertson Method

An iterative approach that alternatively updates the camera response curve and radiance estimates until convergence.

### Mitsunaga & Nayar Method

Uses polynomial modeling of the inverse response function without requiring precise exposure values, capable of self-calibrating the relative exposure values.

### Fast Bilateral Filter Tone Mapping

Implements an efficient bilateral filter for edge-preserving smoothing, preserving details while compressing the dynamic range.

## Results

Output files are stored in the result directory:

-   `hdr_image.hdr` - Raw HDR image
-   `result.jpg` - Tone-mapped result for display
-   `response_curve.jpg` - Plot of recovered camera response curves
-   `original_images.jpg` - Visualization of input images (may be aligned or not)
