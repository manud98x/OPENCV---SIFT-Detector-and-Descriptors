# OpenCV--SIFT-Detector-and-Descriptors

This repository contains Python scripts for SIFT (Scale-Invariant Feature Transform) keypoint detection and image comparison. The implementation includes resizing images to VGA resolution while maintaining aspect ratio, extracting the Y channel, detecting keypoints, and comparing images based on their SIFT descriptors.

## Features

- **Image Resizing**: Resize images to VGA resolution (480x600) while maintaining aspect ratio.
- **SIFT Keypoint Detection**: Detect keypoints and compute descriptors using the SIFT algorithm.
- **Keypoint Visualization**: Draw keypoints on images with orientation lines and cross marks.
- **Descriptor Histograms**: Compute histograms of descriptors and calculate dissimilarity distances.
- **Image Comparison**: Compare images using SIFT descriptors and display a dissimilarity matrix.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
