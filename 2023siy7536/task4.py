from task1 import *
from task2 import *
from task3 import *


def extract_image_features(image):
    contrast_adjusted_image = adjust_contrast(image, alpha=2, beta=22)
    
    blurred_image = gaussian_blur(contrast_adjusted_image, kernel_size=(5, 5), sigma=1.05)
    
    edges = custom_canny_edge_detector(blurred_image, low_threshold=90, high_threshold=180)
    
    dilated = custom_dilate(edges, kernel_size=(3, 3), iterations=1)
    
    bin_image = convert_to_binary(dilated)
    
    contours = find_contours(bin_image)
    transposed_contours = [transpose_contour(contour) for contour in contours]
    height, width = edges.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb = draw_contours(rgb, transposed_contours, color=(0, 255, 0), thickness=1)

    suture_count = len(transposed_contours)

    centroids = calculate_centroids(transposed_contours)
    distances = [compute_intersuture_distance(centroids[i], centroids[i+1], image.shape[0]) for i in range(len(centroids)-1)]

    mean_distance = np.mean(distances)
    var_distance = np.var(distances)

    ref_point_list = []
    for i in range(len(centroids)):
        ref_point = find_reference_point(centroids[i], transposed_contours[i])
        ref_point_list.append(ref_point)
    
    angle_list = []
    for i in range(len(centroids)):
        angle = calculate_angulation(centroids[i], ref_point_list[i])
        angle_list.append(angle)

    angle_mean = np.mean(np.abs(angle_list))
    angle_var = np.var(np.abs(angle_list))

    return suture_count, mean_distance, var_distance, angle_mean, angle_var