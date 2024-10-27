import math

def find_reference_point(centroid, contour_points):
    right_most_point = max(contour_points, key=lambda p: p[0])
    return tuple(right_most_point)

def calculate_angulation(centroid, ref_point):
    xc, yc = centroid
    xr, yr = ref_point

    dy = yr-yc
    dx = xr-xc

    theta = math.atan(dy/(dx+0.00001))
    return math.degrees(theta)
