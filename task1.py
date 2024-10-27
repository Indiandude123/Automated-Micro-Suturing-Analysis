import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import random
# import os
# import csv

def adjust_contrast(image, alpha=1.0, beta=0):
  result_image = np.zeros(image.shape, image.dtype)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      result_image[i,j] = np.clip(alpha*image[i,j] + beta, 0, 255)
  
  return result_image

def custom_convolve2d(image, kernel, mode='same', boundary='symm'):
    """
    Perform 2D convolution on an image with a given kernel.

    Parameters:
    - image: Input image as a NumPy array.
    - kernel: Convolution kernel as a NumPy array.
    - mode: Padding mode ('valid', 'same', or 'full'). Default is 'same'.
    - boundary: Boundary handling ('symm', 'wrap', or 'fill'). Default is 'symm'.

    Returns:
    - result: Convolved image.
    """

    if mode not in ('valid', 'same', 'full'):
        raise ValueError("Invalid mode. Use 'valid', 'same', or 'full'.")

    if boundary not in ('symm', 'wrap', 'fill'):
        raise ValueError("Invalid boundary mode. Use 'symm', 'wrap', or 'fill'.")

    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    if mode == 'valid':
        result_height, result_width = image_height - kernel_height + 1, image_width - kernel_width + 1
    elif mode == 'same':
        result_height, result_width = image_height, image_width
    elif mode == 'full':
        result_height, result_width = image_height + kernel_height - 1, image_width + kernel_width - 1

    result = np.zeros((result_height, result_width), dtype=float)

    # Pad the image based on the boundary mode
    if boundary == 'symm':
        image_padded = np.pad(image, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1)), mode='symmetric')
    elif boundary == 'wrap':
        image_padded = np.pad(image, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1)), mode='wrap')
    elif boundary == 'fill':
        image_padded = np.pad(image, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1)), mode='constant')

    # Perform convolution
    for i in range(result_height):
        for j in range(result_width):
            region = image_padded[i:i + kernel_height, j:j + kernel_width]
            result[i, j] = np.sum(region * kernel)

    return result

def gaussian_blur(image, kernel_size=(7, 7), sigma=1):
    """
    Apply Gaussian blur to an image.

    Parameters:
    - image: Input image as a NumPy array.
    - kernel_size: Size of the Gaussian kernel as a tuple (width, height). Default is (7, 7).
    - sigma: Standard deviation of the Gaussian kernel. Default is 1.

    Returns:
    - blurred_image: Output image after applying Gaussian blur.
    """

    if len(image.shape) == 2:
        # Grayscale image
        kernel = np.fromfunction(
            lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size[0] - 1) / 2) ** 2 + (y - (kernel_size[1] - 1) / 2) ** 2) / (2 * sigma ** 2)),
            kernel_size
        )

        # Normalize the kernel
        kernel = kernel / np.sum(kernel)

        # Apply convolution using the created kernel
        blurred_image = np.zeros_like(image, dtype=float)
        blurred_image = custom_convolve2d(image, kernel, mode='same', boundary='symm')

    elif len(image.shape) == 3:
        # Color image
        blurred_image = np.zeros_like(image, dtype=float)

        # Iterate over each channel
        for channel in range(image.shape[2]):
            kernel = np.fromfunction(
                lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size[0] - 1) / 2) ** 2 + (y - (kernel_size[1] - 1) / 2) ** 2) / (2 * sigma ** 2)),
                kernel_size
            )

            # Normalize the kernel
            kernel = kernel / np.sum(kernel)

            # Apply convolution using the created kernel
            blurred_image[:, :, channel] = custom_convolve2d(image[:, :, channel], kernel, mode='same', boundary='symm')

    return blurred_image.astype(np.uint8)


def custom_canny_edge_detector(image, low_threshold, high_threshold):
    """
    Custom implementation of the Canny edge detector.

    Parameters:
    - image: Input image as a NumPy array (grayscale).
    - low_threshold: Low threshold for edge detection.
    - high_threshold: High threshold for edge detection.

    Returns:
    - edges: Binary image indicating edges.
    """

    # # Step 1: Apply Gaussian blur to the image
    # blurred_image = gaussian_blur(image, kernel_size=(5, 5), sigma=1)

    # Step 2: Calculate gradients using Sobel operators
    gradient_x = custom_convolve2d(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    gradient_y = custom_convolve2d(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    # Step 3: Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Step 4: Non-maximum suppression
    suppressed_magnitude = non_max_suppression(gradient_magnitude, gradient_direction)

    # Step 5: Edge tracking by hysteresis
    edges = hysteresis_edge_tracking(suppressed_magnitude, low_threshold, high_threshold)

    return edges

# Additional functions for Canny edge detection
def non_max_suppression(gradient_magnitude, gradient_direction):
    suppressed_magnitude = gradient_magnitude.copy()

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]

            q = 255
            r = 255

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] < q) or (gradient_magnitude[i, j] < r):
                suppressed_magnitude[i, j] = 0

    return suppressed_magnitude

def hysteresis_edge_tracking(suppressed_magnitude, low_threshold, high_threshold):
    strong_edges = (suppressed_magnitude > high_threshold)
    weak_edges = (suppressed_magnitude >= low_threshold) & (suppressed_magnitude <= high_threshold)

    # Perform edge tracking
    edges = strong_edges.copy()

    for i in range(1, suppressed_magnitude.shape[0] - 1):
        for j in range(1, suppressed_magnitude.shape[1] - 1):
            if weak_edges[i, j]:
                region = suppressed_magnitude[i-1:i+2, j-1:j+2]
                if np.any(region > high_threshold):
                    edges[i, j] = True
                else:
                    edges[i, j] = False

    return edges


def custom_dilate(image, kernel_size, iterations=1):
    """
    Custom implementation of image dilation.

    Parameters:
    - image: Binary image as a NumPy array.
    - kernel_size: Size of the dilation kernel as a tuple (height, width).
    - iterations: Number of iterations. Default is 1.

    Returns:
    - dilated_image: Dilated binary image.
    """

    kernel_height, kernel_width = kernel_size
    dilated_image = image.copy()

    for _ in range(iterations):
        for i in range(kernel_height // 2, image.shape[0] - kernel_height // 2):
            for j in range(kernel_width // 2, image.shape[1] - kernel_width // 2):
                region = image[i - kernel_height // 2:i + kernel_height // 2 + 1, j - kernel_width // 2:j + kernel_width // 2 + 1]
                dilated_image[i, j] = np.max(region)

    return dilated_image


def draw_contours(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contours on an RGB image.

    Parameters:
    - image: RGB image on which contours will be drawn.
    - contours: List of contours, where each contour is a list of (x, y) points.
    - color: Color of the contours in (B, G, R) format. Default is green (0, 255, 0).
    - thickness: Thickness of the contour lines. Default is 2.

    Returns:
    - image: Image with contours drawn.
    """

    # Convert the input image to a NumPy array
    image_array = np.array(image)

    # Iterate through each contour
    for contour in contours:
        # Convert list of (x, y) points to NumPy array
        contour_array = np.array(contour)
        
        # Draw lines connecting the points of the contour
        for i in range(len(contour_array) - 1):
            start_point = tuple(contour_array[i])
            end_point = tuple(contour_array[i + 1])
            cv2.line(image_array, start_point, end_point, color, thickness)

        # Connect the last point to the first point to close the contour
        start_point = tuple(contour_array[-1])
        end_point = tuple(contour_array[0])
        cv2.line(image_array, start_point, end_point, color, thickness)

    return image_array


def convert_to_binary(image):
  result_image = np.zeros(image.shape, image.dtype)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if image[i, j]>0:
        result_image[i,j] = 1
      else:
        result_image[i,j] = 0
  return result_image

def visualize_binary_image(bin_img):
  visual = np.zeros(bin_img.shape, bin_img.dtype)
  for i in range(visual.shape[0]):
    for j in range(visual.shape[1]):
      if bin_img[i,j]==1:
        visual[i,j]=255
  plt.imshow(visual, cmap="gray")
  plt.show()
  
def find_contours(edge_image):
    contours = []

    height, width = edge_image.shape
    visited = np.zeros_like(edge_image, dtype=bool)

    for i in range(height):
        for j in range(width):
            if edge_image[i, j] !=0 and not visited[i, j]:
                contour = trace_contour(edge_image, visited, i, j)
                if contour:
                    contours.append(np.array(contour))
    return contours

def trace_contour(image, visited, start_i, start_j):
    contour = []
    stack = [(start_i, start_j)] 

    while stack:
        current_i, current_j = stack.pop() 
        visited[current_i, current_j] = True
        contour.append((current_i, current_j))

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = current_i + di, current_j + dj

                if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and image[ni, nj] != 0 and not visited[ni, nj]:
                    stack.append((ni, nj))

    return contour

def transpose_contour(contour):
    transposed_contour = np.array([(y, x) for x, y in contour])
    return transposed_contour






  
  