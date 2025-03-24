# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import cv2
from libc.math cimport sqrt, sin, cos, fabs, atan2, M_PI
from libc.stdlib cimport malloc, free, rand, RAND_MAX

# Define numpy types
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.float64_t DTYPE_FLOAT64_t
ctypedef np.float32_t DTYPE_FLOAT32_t

cdef struct Point:
    int x
    int y

cdef struct Line:
    int x1
    int y1
    int x2
    int y2
    int votes

cdef struct Circle:
    int x
    int y
    int radius
    int votes

cdef struct Ellipse:
    int x
    int y
    int a  # semi-major axis
    int b  # semi-minor axis
    double angle
    int votes

def cy_detect_lines(np.ndarray[DTYPE_UINT8_t, ndim=2] edges, 
                    np.ndarray original_image,
                    int threshold,
                    double rho=1.0, 
                    double theta=M_PI/180, 
                    int min_line_length=30, 
                    int max_line_gap=10):
    """
    Line detection using Hough transform implemented from scratch
    
    Parameters:
        edges: Binary edge image
        original_image: Original image to draw lines on
        threshold: Minimum number of votes to be considered a line
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        min_line_length: Minimum line length
        max_line_gap: Maximum allowed gap between line segments
    """
    # Create a copy of the original image for output
    cdef np.ndarray output_image = original_image.copy()
    cdef int height = edges.shape[0]
    cdef int width = edges.shape[1]
    
    # Calculate the maximum distance in the image diagonal
    cdef int max_distance = <int>(sqrt(width*width + height*height))
    
    # Define the accumulator dimensions
    cdef int num_thetas = <int>(M_PI / theta)
    cdef int num_rhos = <int>(2 * max_distance / rho)
    
    # Create the accumulator array
    cdef np.ndarray[DTYPE_INT32_t, ndim=2] accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    
    # Create a map to store edge pixels
    cdef list edge_points = []
    cdef int x, y, i, j, r_idx, t_idx
    cdef double r
    
    # Find all edge points
    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:
                edge_points.append((x, y))
    
    # Vote in the accumulator array
    cdef double cos_theta, sin_theta
    for x, y in edge_points:
        for t_idx in range(num_thetas):
            # Calculate theta
            theta_val = t_idx * theta
            cos_theta = cos(theta_val)
            sin_theta = sin(theta_val)
            
            # Calculate rho
            r = x * cos_theta + y * sin_theta
            
            # Convert rho to index in accumulator (shift to make all values positive)
            r_idx = <int>((r + max_distance) / rho)
            
            # Increment the accumulator
            if 0 <= r_idx < num_rhos:
                accumulator[r_idx, t_idx] += 1
    
    # Find peaks in the accumulator (values above threshold)
    cdef list lines = []
    cdef int line_count = 0
    cdef int x1, y1, x2, y2, count
    
    for r_idx in range(num_rhos):
        for t_idx in range(num_thetas):
            if accumulator[r_idx, t_idx] >= threshold:
                # Convert back from accumulator space to image space
                r = (r_idx * rho) - max_distance
                theta_val = t_idx * theta
                
                # Calculate line endpoints
                # For a vertical line (theta near 0 or PI)
                if fabs(sin(theta_val)) < 0.01:
                    # x = r/cos(theta)
                    x1 = x2 = <int>(r / cos(theta_val))
                    y1 = 0
                    y2 = height - 1
                else:
                    # y = (r - x*cos(theta))/sin(theta)
                    x1 = 0
                    y1 = <int>(r / sin(theta_val))
                    x2 = width - 1
                    y2 = <int>((r - x2 * cos(theta_val)) / sin(theta_val))
                
                # Calculate line length
                length = sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Check if line meets minimum length requirement
                if length >= min_line_length:
                    # Draw the line
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
                    line_count += 1
    
    return output_image, line_count

def cy_detect_circles(np.ndarray[DTYPE_UINT8_t, ndim=2] gray_image,
                      np.ndarray[DTYPE_UINT8_t, ndim=2] edges,
                      np.ndarray original_image,
                      int threshold, 
                      int min_dist=20, 
                      int param1=50,
                      int min_radius=10, 
                      int max_radius=100):
    """
    Circle detection using Hough transform implemented from scratch
    
    Parameters:
        gray_image: Grayscale input image
        original_image: Original image to draw circles on
        threshold: Minimum number of votes to be considered a circle
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for Canny edge detector
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    """
    # Create a copy of the original image for output
    cdef np.ndarray output_image = original_image.copy()
    cdef int height = gray_image.shape[0]
    cdef int width = gray_image.shape[1]
    
    # Extract edges using Canny
    # cdef np.ndarray[DTYPE_UINT8_t, ndim=2] edges = cv2.Canny(gray_image, param1 // 2, param1)
    
    # Create a 3D accumulator array (x, y, radius)
    cdef np.ndarray[DTYPE_INT32_t, ndim=3] accumulator = np.zeros((width, height, max_radius - min_radius + 1), dtype=np.int32)
    
    # Create a map to store edge pixels
    cdef list edge_points = []
    cdef int x, y, a, b, r, r_idx
    
    # Find all edge points
    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:
                edge_points.append((x, y))
    
    # Vote in the accumulator array
    cdef double angle, x_center, y_center
    for x, y in edge_points:
        # For each edge point, vote for all possible circles
        for r in range(min_radius, max_radius + 1):
            r_idx = r - min_radius
            
            # Optimize by voting at a subset of angles for each radius
            for angle_idx in range(0, 360, 15):  # Step by 15 degrees
                angle = angle_idx * M_PI / 180.0
                
                # Calculate possible circle center
                a = <int>(x - r * cos(angle))
                b = <int>(y - r * sin(angle))
                
                # Check if the center is within image bounds
                if 0 <= a < width and 0 <= b < height:
                    accumulator[a, b, r_idx] += 1
    
    # Find peaks in the accumulator (circles above threshold)
    cdef list circles = []
    cdef int circle_count = 0
    cdef int center_x, center_y, radius, score
    cdef bint is_duplicate
    
    # Extract circle candidates
    for y in range(height):
        for x in range(width):
            for r_idx in range(max_radius - min_radius + 1):
                if accumulator[x, y, r_idx] >= threshold:
                    radius = min_radius + r_idx
                    circles.append((x, y, radius, accumulator[x, y, r_idx]))
    
    # Sort circles by score (number of votes)
    circles.sort(key=lambda c: c[3], reverse=True)
    
    # Non-maximum suppression
    cdef list final_circles = []
    for circle in circles:
        center_x, center_y, radius, score = circle
        
        # Check if this circle is a duplicate (close to an existing one)
        is_duplicate = False
        for fc in final_circles:
            fc_x, fc_y, fc_r, _ = fc
            dist = sqrt((center_x - fc_x)**2 + (center_y - fc_y)**2)
            if dist < min_dist:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_circles.append(circle)
            # Draw the circle
            cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 2, cv2.LINE_AA)
            # Draw the center
            cv2.circle(output_image, (center_x, center_y), 2, (0, 0, 255), 3, cv2.LINE_AA)
            circle_count += 1
    
    return output_image, circle_count

def cy_detect_ellipses(np.ndarray[DTYPE_UINT8_t, ndim=2] edges,
                        np.ndarray original_image,
                        int threshold,
                        int min_area=100,
                        int max_area=10000,
                        int min_points=5):
    """
    Simplified and robust ellipse detection with relaxed constraints
    
    Parameters:
        edges: Binary edge image
        original_image: Original image to draw ellipses on
        threshold: Minimum contour area to be considered
        min_area: Minimum ellipse area
        max_area: Maximum ellipse area
        min_points: Minimum number of points for ellipse estimation
    """
    # Create a copy of the original image for output
    cdef np.ndarray output_image = original_image.copy()
    cdef int height = edges.shape[0]
    cdef int width = edges.shape[1]
    
    # Make sure we have a binary image for contour finding
    cdef np.ndarray[DTYPE_UINT8_t, ndim=2] edges_copy = edges.copy()
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, try with different retrieval mode
    if len(contours) == 0:
        contours, _ = cv2.findContours(edges_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cdef int ellipse_count = 0
    cdef double area, ellipse_area, aspect_ratio, quality
    cdef tuple center, axes
    cdef double angle
    
    # Process each contour
    for contour in contours:
        # Check if contour has enough points to fit an ellipse
        if len(contour) >= min_points:  
            # Calculate contour area for initial filtering
            area = cv2.contourArea(contour)
            
            # Apply a more relaxed area threshold - use threshold as minimum size
            if area < threshold / 2:  # More lenient threshold
                continue
                
            try:
                # Try to fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                
                # Calculate ellipse area
                ellipse_area = M_PI * axes[0] * axes[1] / 4
                
                # Calculate aspect ratio
                aspect_ratio = max(axes) / (min(axes) + 1e-5)  # Avoid division by zero
                
                # Filter based on quality metrics - more relaxed constraints
                if (center[0] >= 0 and center[0] < width and
                    center[1] >= 0 and center[1] < height and
                    ellipse_area > threshold / 4 and  # Very relaxed minimum area
                    ellipse_area < max_area * 2 and  # More relaxed maximum area
                    aspect_ratio < 10.0):  # More relaxed aspect ratio
                    
                    # Evaluate ellipse fit quality
                    quality = simple_ellipse_fit(contour, ellipse)
                    
                    # Use a lower quality threshold
                    if quality > 0.5:  # More lenient quality threshold
                        # Draw the ellipse
                        cv2.ellipse(output_image, ellipse, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # Draw the center
                        cv2.circle(output_image, 
                                (int(center[0]), int(center[1])), 
                                3, (0, 0, 255), -1)
                        
                        ellipse_count += 1
            except Exception as e:
                # Print the error for debugging
                # print(f"Ellipse fitting error: {e}")
                continue
    
    
    return output_image, ellipse_count

cdef double simple_ellipse_fit(contour, tuple ellipse):
    """
    Simplified evaluation of how well an ellipse fits a contour
    """
    center, axes, angle = ellipse
    cx, cy = center
    a, b = axes[0]/2, axes[1]/2  # Semi-major and semi-minor axes
    angle_rad = angle * M_PI / 180.0
    
    cdef int num_points = len(contour)
    cdef int count_good_points = 0
    cdef double x, y, dist
    cdef double tolerance = 3.0  # More generous tolerance
    
    # Sample fewer points for efficiency
    cdef int step = max(1, num_points // 20)  # Sample at most 20 points
    
    for i in range(0, num_points, step):
        x, y = contour[i][0][0], contour[i][0][1]
        
        # Calculate normalized distance to ellipse
        dist = simple_distance_to_ellipse(x, y, cx, cy, a, b, angle_rad)
        
        # Use a more lenient criterion
        if dist < tolerance:
            count_good_points += 1
    
    return count_good_points / (num_points / step + 1)

cdef double simple_distance_to_ellipse(double x, double y, double cx, double cy, 
                                     double a, double b, double angle):
    """
    Simplified distance calculation - approximates distance to ellipse
    Returns absolute difference from the normalized ellipse equation
    (smaller is better, 0 means exactly on the ellipse)
    """
    # Translate point to ellipse center
    cdef double tx = x - cx
    cdef double ty = y - cy
    
    # Rotate to align with ellipse axes
    cdef double cos_angle = cos(angle)
    cdef double sin_angle = sin(angle)
    cdef double x_rot = tx * cos_angle + ty * sin_angle
    cdef double y_rot = -tx * sin_angle + ty * cos_angle
    
    # Calculate normalized distance to ellipse (0 means exactly on the ellipse)
    cdef double normalized = (x_rot * x_rot) / (a * a) + (y_rot * y_rot) / (b * b)
    
    # Return absolute difference from 1.0 (which represents the ellipse boundary)
    return fabs(normalized - 1.0)