import numpy as np
import cv2



def calculate_moments(contour):
    """
    Calculate moments for a given contour.

    Parameters:
    - contour: List of (x, y) coordinates representing the contour.

    Returns:
    - moments: Dictionary containing calculated moments.
    """
    moments = {'m00': 0, 'm01': 0, 'm10': 0}
    for point in contour:
        x, y = point
        moments['m00'] += 1
        moments['m01'] += y
        moments['m10'] += x
    return moments

def calculate_centroid(moments):
    """
    Calculate the centroid coordinates using moments.

    Parameters:
    - moments: Dictionary containing calculated moments.

    Returns:
    - centroid: Tuple (x, y) representing the centroid coordinates.
    """
    if moments['m00'] != 0:
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        return (centroid_x, centroid_y)
    else:
        return None

def calculate_centroids(contours):
    """
    Calculate centroids for a list of contours.

    Parameters:
    - contours: List of contours, where each contour is represented as a list of (x, y) coordinates.

    Returns:
    - centroids: List of tuples representing the centroids for each contour.
    """
    centroids = []
    for contour in contours:
        moments = calculate_moments(contour)
        centroid = calculate_centroid(moments)
        if centroid is not None:
            centroids.append(centroid)
    return centroids



def compute_intersuture_distance(pt1, pt2, image_height):
    # (x1-x2)**2 + (y1-y2)**2
    x1, y1 = pt1
    x2, y2 = pt2
    euclid_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    intersuture_dist = euclid_dist/image_height

    return intersuture_dist

def visualize_distances(image, centroids, distances):
    """
    Visualize distances and centroids on the original image.

    Parameters:
    - image: Original image.
    - centroids: List of tuples representing centroids (x, y).
    - distances: List of distances between consecutive centroids.

    Returns:
    - image_with_visualization: Image with distances and centroids visualized.
    """
    image_with_visualization = image.copy()

    for i in range(len(centroids)-1):
        cv2.line(image_with_visualization, tuple(map(int, centroids[i])), tuple(map(int, centroids[i+1])), (0, 255, 0), 2)
        cv2.putText(image_with_visualization, f"{distances[i]:.2f}", tuple(map(int, np.mean([centroids[i], centroids[i+1]], axis=0))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return image_with_visualization