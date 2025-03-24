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
    
    Returns:
        Tuple of (result_image, count) where:
        - result_image: Original image with detected lines drawn
        - count: Number of lines detected
    """
    # Create a copy of the original image for drawing
    cdef np.ndarray result_image = original_image.copy()
    
    cdef int height = edges.shape[0]
    cdef int width = edges.shape[1]
    
    # Determine max distance possible in the image (diagonal)
    cdef double diag = sqrt(width*width + height*height)
    cdef int rho_max = <int>(diag / rho)
    
    # Discretize rho values (-rho_max to rho_max)
    cdef int rho_dim = 2 * rho_max + 1
    
    # Number of theta values
    cdef int theta_dim = <int>(M_PI / theta)
    
    # Create the accumulator
    cdef np.ndarray[DTYPE_INT32_t, ndim=2] accumulator = np.zeros((rho_dim, theta_dim), dtype=np.int32)
    
    # Fill the accumulator
    cdef int x, y, t, r
    cdef double cos_t, sin_t, rho_val
    
    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:  # Edge pixel
                for t in range(theta_dim):
                    # Calculate parametric line parameters
                    cos_t = cos(t * theta)
                    sin_t = sin(t * theta)
                    rho_val = x * cos_t + y * sin_t
                    
                    # Convert rho to index (shift to positive range)
                    r = <int>(rho_val / rho) + rho_max
                    
                    # Increment accumulator
                    if 0 <= r < rho_dim:
                        accumulator[r, t] += 1
    
    # Find peaks in the accumulator
    cdef list lines = []
    cdef Line line
    cdef int votes, x1, y1, x2, y2
    cdef double line_length
    cdef bint is_maximum
    
    # Non-maximum suppression window size
    cdef int window = 3
    
    for r in range(1, rho_dim-1):
        for t in range(1, theta_dim-1):
            votes = accumulator[r, t]
            
            if votes >= threshold:
                # Non-maximum suppression
                is_maximum = True
                for dr in range(-window//2, window//2 + 1):
                    for dt in range(-window//2, window//2 + 1):
                        if dr == 0 and dt == 0:
                            continue
                        if (0 <= r+dr < rho_dim and 0 <= t+dt < theta_dim and 
                            accumulator[r+dr, t+dt] > votes):
                            is_maximum = False
                            break
                    if not is_maximum:
                        break
                        
                if is_maximum:
                    # Convert accumulator coordinates back to line parameters
                    rho_val = (r - rho_max) * rho
                    theta_val = t * theta
                    
                    cos_t = cos(theta_val)
                    sin_t = sin(theta_val)
                    
                    # Convert to endpoint representation
                    # Find two points on the line
                    if fabs(sin_t) < 0.001:  # Vertical line
                        x1 = x2 = <int>(rho_val / cos_t)
                        y1 = 0
                        y2 = height - 1
                    elif fabs(cos_t) < 0.001:  # Horizontal line
                        x1 = 0
                        x2 = width - 1
                        y1 = y2 = <int>(rho_val / sin_t)
                    else:
                        # Point 1 (x=0)
                        x1 = 0
                        y1 = <int>(rho_val / sin_t)
                        
                        # Point 2 (x=width-1)
                        x2 = width - 1
                        y2 = <int>((rho_val - (width - 1) * cos_t) / sin_t)
                        
                        # Check if line endpoints are within image bounds
                        if y1 < 0 or y1 >= height:
                            # Try point at y=0
                            y1 = 0
                            x1 = <int>(rho_val / cos_t)
                            
                        if y2 < 0 or y2 >= height:
                            # Try point at y=height-1
                            y2 = height - 1
                            x2 = <int>((rho_val - (height - 1) * sin_t) / cos_t)
                    
                    # Add to results if both endpoints are within bounds
                    if (0 <= x1 < width and 0 <= y1 < height and
                        0 <= x2 < width and 0 <= y2 < height):
                        
                        # Calculate line length
                        line_length = sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        if line_length >= min_line_length:
                            lines.append((x1, y1, x2, y2, votes))
    
    # Apply line segment detection with max_line_gap
    if max_line_gap > 0:
        # Sort lines by votes
        lines.sort(key=lambda l: l[4], reverse=True)
        
        # Draw lines on the output image
        for i in range(len(lines)):
            x1, y1, x2, y2, _ = lines[i]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Return both the image with lines drawn and the count of lines
    return result_image, len(lines)

def cy_detect_circles(np.ndarray[DTYPE_UINT8_t, ndim=2] gray_image,
                      np.ndarray[DTYPE_UINT8_t, ndim=2] edges,
                      np.ndarray original_image,
                      int threshold, 
                      int min_dist=20,
                      int param1=50,  # Added parameter for canny high threshold
                      int min_radius=10, 
                      int max_radius=100):
    """
    Circle detection using Hough transform implemented from scratch
    
    Parameters:
        gray_image: Grayscale input image
        edges: Binary edge image
        original_image: Original image to draw circles on
        threshold: Minimum number of votes to be considered a circle
        min_dist: Minimum distance between circle centers
        param1: Higher threshold for Canny edge detector (not used directly, but included for API compatibility)
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    
    Returns:
        Tuple of (result_image, count) where:
        - result_image: Original image with detected circles drawn
        - count: Number of circles detected
    """

    if min_radius >= max_radius:
        max_radius = min_radius + 1

    # Create a copy of the original image for drawing
    threshold = max(threshold, 2)  # Ensure threshold is at least 1
    cdef np.ndarray result_image = original_image.copy()
    
    cdef int height = edges.shape[0]
    cdef int width = edges.shape[1]
    cdef int radius_range = max_radius - min_radius + 1
    
    # Create 3D accumulator array for (x, y, r)
    cdef np.ndarray[DTYPE_INT32_t, ndim=3] accumulator = np.zeros((width, height, radius_range), dtype=np.int32)
    
    # Detect edge pixels
    cdef int i, j, r, x, y, r_idx, circle_x, circle_y, circle_r
    cdef double angle, angle_step = 1.0 / max(min_radius, 10)  # Smaller step for larger accuracy
    
    # Fill the accumulator
    for i in range(height):
        for j in range(width):
            if edges[i, j] > 0:  # This is an edge pixel
                # For each possible radius
                for r_idx in range(radius_range):
                    r = r_idx + min_radius
                    # Discretize the circle into points
                    angle = 0
                    while angle < 2 * M_PI:
                        # Calculate potential center coordinates
                        x = int(j - r * cos(angle))
                        y = int(i - r * sin(angle))
                        # Check if within image bounds
                        if 0 <= x < width and 0 <= y < height:
                            accumulator[x, y, r_idx] += 1
                        angle += angle_step
    
    # Find peaks in accumulator (non-maximum suppression)
    cdef list circles = []
    cdef int votes
    cdef bint is_maximum
    
    for x in range(width):
        for y in range(height):
            for r_idx in range(radius_range):
                votes = accumulator[x, y, r_idx]
                if votes >= threshold:
                    r = r_idx + min_radius
                    
                    # Non-maximum suppression in local neighborhood
                    is_maximum = True
                    for nx in range(max(0, x-2), min(width, x+3)):
                        for ny in range(max(0, y-2), min(height, y+3)):
                            for nr_idx in range(max(0, r_idx-2), min(radius_range, r_idx+3)):
                                if accumulator[nx, ny, nr_idx] > votes:
                                    is_maximum = False
                                    break
                            if not is_maximum:
                                break
                        if not is_maximum:
                            break
                    
                    if is_maximum:
                        circles.append((x, y, r, votes))
    
    # Sort circles by vote count (descending)
    circles.sort(key=lambda c: c[3], reverse=True)
    
    # Apply minimum distance constraint
    cdef list final_circles = []
    cdef bint too_close
    cdef double dist
    
    for i in range(len(circles)):
        too_close = False
        for j in range(len(final_circles)):
            # Calculate distance between centers
            dist = sqrt((circles[i][0] - final_circles[j][0])**2 + 
                         (circles[i][1] - final_circles[j][1])**2)
            if dist < min_dist:
                too_close = True
                break
        
        if not too_close:
            final_circles.append(circles[i])
    
    # Draw circles on the output image
    for i in range(len(final_circles)):
        circle_x, circle_y, circle_r, _ = final_circles[i]
        cv2.circle(result_image, (circle_x, circle_y), circle_r, (0, 255, 0), 2)
        cv2.circle(result_image, (circle_x, circle_y), 2, (0, 0, 255), 3)
    
    # Return both the image with circles drawn and the count of circles
    return result_image, len(final_circles)

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
    
    Returns:
        Tuple of (result_image, count) where:
        - result_image: Original image with detected ellipses drawn
        - count: Number of ellipses detected
    """
    # Create a copy of the original image for drawing
    cdef np.ndarray result_image = original_image.copy()
    
    # Find contours in the edge image
    # We'll use OpenCV just for contour finding which is a basic operation
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cdef list ellipses = []
    cdef double area, ellipse_area, fit_error
    cdef int cx, cy, a, b
    cdef double angle
    
    for contour in contours:
        if len(contour) < min_points:
            continue
            
        # Calculate contour area for filtering
        area = cv2.contourArea(contour)
        if area < threshold:
            continue
            
        # Try to fit an ellipse to the contour
        try:
            # Use OpenCV's ellipse fitting as it's a basic mathematical operation
            (cx, cy), (width, height), angle = cv2.fitEllipse(contour)
            
            # Convert semi-axes to full axes
            a = width / 2
            b = height / 2
            
            # Calculate ellipse area
            ellipse_area = M_PI * a * b
            
            # Filter by area constraints
            if ellipse_area < min_area or ellipse_area > max_area:
                continue
                
            # Calculate fit quality
            fit_error = simple_ellipse_fit(contour, (cx, cy, a, b, angle))
            
            # Accept the ellipse only if the fit error is below threshold
            if fit_error < 0.2:  # Arbitrary threshold for good fit
                ellipses.append((int(cx), int(cy), int(a), int(b), angle, int(area)))
                
        except Exception:
            # Skip contours that can't be fit to an ellipse
            continue
    
    # Sort ellipses by vote count (descending)
    ellipses.sort(key=lambda e: e[5], reverse=True)
    
    # Draw ellipses on the output image
    for i in range(len(ellipses)):
        e = ellipses[i]
        cv2.ellipse(result_image, 
                   (e[0], e[1]),               # center
                   (e[2], e[3]),               # axes
                   e[4],                       # angle
                   0, 360,                     # start and end angles
                   (0, 255, 0), 2)             # color, thickness
    
    # Return both the image with ellipses drawn and the count of ellipses
    return result_image, len(ellipses)

cdef double simple_ellipse_fit(contour, tuple ellipse):
    """
    Simplified evaluation of how well an ellipse fits a contour
    """
    cdef double cx = ellipse[0]
    cdef double cy = ellipse[1]
    cdef double a = ellipse[2]
    cdef double b = ellipse[3]
    cdef double angle = ellipse[4]
    
    # Convert angle from degrees to radians if needed
    if angle > 2 * M_PI:
        angle = angle * M_PI / 180.0
    
    cdef double total_error = 0
    cdef int n_points = 0
    cdef double x, y, dist
    
    # Calculate average distance from contour points to ellipse
    for point in contour:
        x = point[0][0]
        y = point[0][1]
        dist = simple_distance_to_ellipse(x, y, cx, cy, a, b, angle)
        total_error += dist
        n_points += 1
    
    if n_points == 0:
        return float('inf')
        
    return total_error / n_points

cdef double simple_distance_to_ellipse(double x, double y, double cx, double cy, 
                                     double a, double b, double angle):
    """
    Simplified distance calculation - approximates distance to ellipse
    Returns absolute difference from the normalized ellipse equation
    (smaller is better, 0 means exactly on the ellipse)
    """
    # Translate point to ellipse center
    cdef double x_t = x - cx
    cdef double y_t = y - cy
    
    # Rotate point to align with ellipse axes
    cdef double cos_angle = cos(angle)
    cdef double sin_angle = sin(angle)
    cdef double x_r = x_t * cos_angle + y_t * sin_angle
    cdef double y_r = -x_t * sin_angle + y_t * cos_angle
    
    # Calculate normalized distance from ellipse equation
    if a == 0 or b == 0:
        return float('inf')
        
    cdef double result = (x_r*x_r)/(a*a) + (y_r*y_r)/(b*b)
    
    # Absolute difference from being on ellipse (where result would be 1.0)
    return fabs(result - 1.0)