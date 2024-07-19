# OpenCV--SIFT-Detector-and-Descriptors

# SIFT Image Processing and Comparison

This repository contains Python scripts for performing SIFT (Scale-Invariant Feature Transform) keypoint detection and descriptor extraction on images, resizing images to VGA resolution, and comparing multiple images based on SIFT descriptors using histogram distance.

## Table of Contents

- [Introduction](#introduction)
- [Tasks](#tasks)
  - [Task 1: Image Resizing and SIFT Keypoint Detection](#task-1-image-resizing-and-sift-keypoint-detection)
  - [Task 2: SIFT Descriptor Comparison](#task-2-sift-descriptor-comparison)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [License](#license)

## Introduction

This project aims to demonstrate image processing tasks using OpenCV, including resizing images to VGA resolution while maintaining aspect ratio, detecting SIFT keypoints, and comparing multiple images using SIFT descriptors and histogram distance.

## Tasks

### Task 1: Image Resizing and SIFT Keypoint Detection

This task resizes an input image to VGA resolution (480x600) while maintaining its aspect ratio, converts it to the Y channel of the XYZ color space, and detects SIFT keypoints on the Y channel. It also draws keypoints on the resized image.

#### Features

- Resize image to VGA resolution while maintaining aspect ratio.
- Convert image to the Y channel of XYZ color space.
- Detect SIFT keypoints and compute descriptors.
- Draw keypoints on the resized image.

### Task 2: SIFT Descriptor Comparison

This task compares multiple images by detecting SIFT keypoints, extracting descriptors, clustering descriptors using k-means, and computing histogram distances between images to create a dissimilarity matrix.

#### Features

- Detect SIFT keypoints and descriptors for multiple images.
- Perform k-means clustering on descriptors.
- Compute histogram of descriptors for each image.
- Calculate histogram distances and create a dissimilarity matrix.

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Usage

### SIFT Keypoint Detection for a Single Image

```python siftImages.py <image_file>```

### SIFT Image Comparison for Multiple Images

```python siftImages.py <image_file1> <image_file2> ... ```
