# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import cv2
from libc.math cimport sqrt, sin, cos, fabs, atan2, M_PI, pow

# Define numpy types
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.float64_t DTYPE_FLOAT64_t
ctypedef np.float32_t DTYPE_FLOAT32_t

cdef struct Point:
    int x
    int y

def cy_initial_contour(tuple center, int radius, int num_points=1000):
    """
    Generate initial circular contour points using Cython for better performance
    
    Parameters:
        center: (x, y) center coordinates
        radius: radius of the initial circle
        num_points: number of points to generate around the circle
    
    Returns:
        List of (x, y) coordinates representing the initial contour
    """
    cdef list initial_snake = []
    cdef double current_angle = 0.0
    cdef double resolution = 360.0 / num_points
    cdef int i, x_point, y_point
    cdef int cx = center[0]
    cdef int cy = center[1]
    
    # Generate points in a circle using trigonometric functions
    # This avoids using cv2.polarToCart which is causing the dimension error
    for i in range(num_points):
        angle_rad = current_angle * M_PI / 180.0
        x_point = int(cx + radius * cos(angle_rad))
        y_point = int(cy + radius * sin(angle_rad))
        
        initial_snake.append((x_point, y_point))
        current_angle += resolution
    
    return initial_snake

cdef double cy_internal_energy(Point pt, Point prev_pt, Point next_pt, double alpha, double beta):
    """
    Calculate internal energy based on tension and stiffness terms
    exactly as in the Python implementation
    
    Parameters:
        pt: Current point
        prev_pt: Previous point in the contour
        next_pt: Next point in the contour
        alpha: Tension coefficient
        beta: Stiffness coefficient
        
    Returns:
        Total internal energy
    """
    # Elasticity term (first derivative approximation) - Tension
    cdef double tension = alpha * ((pt.x - prev_pt.x)**2 + (pt.y - prev_pt.y)**2)
    
    # Stiffness term (second derivative approximation) - Curvature
    cdef double curvature = beta * ((next_pt.x - 2*pt.x + prev_pt.x)**2 + 
                                    (next_pt.y - 2*pt.y + prev_pt.y)**2)
    
    # Total internal energy
    return tension + curvature

cdef double cy_external_energy(
    np.ndarray[DTYPE_UINT8_t, ndim=2] img,
    np.ndarray[DTYPE_FLOAT64_t, ndim=2] grad_x,
    np.ndarray[DTYPE_FLOAT64_t, ndim=2] grad_y,
    Point pt, 
    double beta
):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef double grad_mag
    
    if 0 <= pt.x < w and 0 <= pt.y < h:
        if grad_x is not None and grad_y is not None:
            # Use normalized gradient magnitude
            grad_mag = sqrt(grad_x[pt.y, pt.x] ** 2 + grad_y[pt.y, pt.x] ** 2)
            # Invert so that high gradients are low energy (attract snake)
            return -beta * grad_mag
        else:
            # Use image intensity
            return -beta * img[pt.y, pt.x]
    else:
        return 1000.0  # High energy to discourage out-of-bounds movement

cdef double cy_balloon_energy(Point pt, Point prev_pt, double gamma):
    """
    Calculate balloon energy with improved directional control
    """
    cdef double dx = pt.x - prev_pt.x
    cdef double dy = pt.y - prev_pt.y
    cdef double dist = sqrt(dx*dx + dy*dy)
    
    # Normalize direction and apply gamma
    if dist > 1e-10:
        # Positive gamma = inflation, Negative gamma = deflation
        return gamma * dist
    else:
        return 0.0

def cy_evolve_contour(
    list snake_points,
    np.ndarray[DTYPE_UINT8_t, ndim=2] img,
    np.ndarray[DTYPE_FLOAT64_t, ndim=2] grad_x,
    np.ndarray[DTYPE_FLOAT64_t, ndim=2] grad_y,
    int iterations,
    double alpha,
    double beta,
    double gamma
):
    """
    Evolve the contour using active contour model (snake)
    
    Parameters:
        snake_points: Initial contour points
        img: Grayscale image
        grad_x: Pre-computed X gradient
        grad_y: Pre-computed Y gradient
        iterations: Number of iterations
        alpha: Internal energy weight
        beta: External energy weight
        gamma: Balloon energy weight
        
    Returns:
        Evolved contour points
    """
    cdef int num_points = len(snake_points)
    cdef list new_curve
    cdef int i, j, dx, dy, iter_count
    cdef double min_energy, energy, internal_e, external_e, balloon_e
    cdef Point pt, prev_pt, next_pt, new_pt, move_pt
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    
    # Optimize: convert python list to C array of Points for faster access
    cdef list current_snake = snake_points.copy()
    
    # Evolve contour through iterations
    for iter_count in range(iterations):
        new_curve = [None] * num_points
        
        for i in range(num_points):
            # Get current point and neighbors
            pt.x, pt.y = current_snake[i]
            prev_pt.x, prev_pt.y = current_snake[(i - 1 + num_points) % num_points]
            next_pt.x, next_pt.y = current_snake[(i + 1) % num_points]
            
            min_energy = float('inf')
            new_pt = pt
            
            # Check neighborhood (3x3)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    move_pt.x = pt.x + dx
                    move_pt.y = pt.y + dy
                    
                    # Check boundaries
                    if not (0 <= move_pt.x < w and 0 <= move_pt.y < h):
                        continue
                    
                    # Calculate energies - passing both alpha and beta to internal energy
                    internal_e = cy_internal_energy(move_pt, prev_pt, next_pt, alpha, beta)
                    external_e = cy_external_energy(img, grad_x, grad_y, move_pt, beta)
                    balloon_e = cy_balloon_energy(move_pt, prev_pt, gamma)
                    
                    # Total energy
                    energy = internal_e + external_e + balloon_e
                    
                    # Keep minimum energy point
                    if energy < min_energy:
                        min_energy = energy
                        new_pt = move_pt
            
            # Store result as a tuple
            new_curve[i] = (new_pt.x, new_pt.y)
        
        # Update snake for next iteration
        current_snake = new_curve
    
    return current_snake

def cy_calculate_metrics(list snake_points):
    """
    Calculate perimeter and area of the contour
    
    Parameters:
        snake_points: List of contour points
        
    Returns:
        (perimeter, area)
    """
    cdef int num_points = len(snake_points)
    cdef double perimeter = 0.0
    cdef int i
    cdef double dx, dy
    cdef tuple current, next_pt
    
    # Calculate perimeter
    for i in range(num_points):
        current = snake_points[i]
        next_pt = snake_points[(i + 1) % num_points]
        dx = current[0] - next_pt[0]
        dy = current[1] - next_pt[1]
        perimeter += sqrt(dx*dx + dy*dy)
    
    # Calculate area using OpenCV
    contour_array = np.array(snake_points, dtype=np.int32)
    area = cv2.contourArea(contour_array)
    
    return perimeter, area

def cy_get_chain_code(list snake_points):
    """
    Calculate the 8-direction chain code of contour
    
    Parameters:
        snake_points: List of contour points
        
    Returns:
        List of chain code directions (0-7)
    """
    cdef int num_points = len(snake_points)
    cdef list chain_code = []
    cdef int i, code
    cdef tuple current, next_point
    cdef int dx, dy
    
    for i in range(num_points):
        current = snake_points[i]
        next_point = snake_points[(i + 1) % num_points]
        
        # Direction vector
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        
        # Normalize direction
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        
        # Convert to 8-direction chain code
        if dx == 0 and dy == -1:      # North
            code = 0
        elif dx == 1 and dy == -1:    # Northeast
            code = 1
        elif dx == 1 and dy == 0:     # East
            code = 2
        elif dx == 1 and dy == 1:     # Southeast
            code = 3
        elif dx == 0 and dy == 1:     # South
            code = 4
        elif dx == -1 and dy == 1:    # Southwest
            code = 5
        elif dx == -1 and dy == 0:    # West
            code = 6
        elif dx == -1 and dy == -1:   # Northwest
            code = 7
        else:
            code = -1  # Invalid direction
        
        chain_code.append(code)
    
    return chain_code

def process_contour_cython(
    np.ndarray[DTYPE_UINT8_t, ndim=2] img_original,
    np.ndarray[DTYPE_UINT8_t, ndim=2] img_grayscale,
    tuple initial_center,
    int initial_radius,
    int iterations=100,
    double alpha=0.01,
    double beta=0.1,
    double gamma=0.05,
    int num_points=100
):
    """
    Process contour using optimized Cython implementation
    
    Parameters:
        img_original: Original image
        img_grayscale: Grayscale image
        initial_center: Initial contour center (x, y)
        initial_radius: Initial contour radius
        iterations: Number of iterations
        alpha: Internal energy weight
        beta: External energy weight
        gamma: Balloon energy weight
        num_points: Number of points in the contour
        
    Returns:
        (contour_points, perimeter, area, chain_code)
    """
    if img_original is None or img_grayscale is None:
        return None, 0, 0, []
    
    # Create initial contour
    snake_points = cy_initial_contour(initial_center, initial_radius, num_points)
    
    # Prepare image for processing - improved preprocessing
    cdef np.ndarray[DTYPE_UINT8_t, ndim=2] gray_img = img_grayscale.copy()
    
    # Apply a more aggressive blur for noisy images
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5)
    
    # Enhance contrast
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Pre-compute gradient images
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=2] gradient_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=2] gradient_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize gradients
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=2] gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    cdef double max_mag = np.max(gradient_magnitude)
    if max_mag > 0:
        gradient_x = gradient_x / max_mag
        gradient_y = gradient_y / max_mag
    
    # Evolve contour
    contour_points = cy_evolve_contour(
        snake_points,
        gray_img,
        gradient_x,
        gradient_y,
        iterations,
        alpha,
        beta,
        gamma
    )
    
    # Calculate metrics
    perimeter, area = cy_calculate_metrics(contour_points)
    
    # Get chain code
    chain_code = cy_get_chain_code(contour_points)
    
    return contour_points, perimeter, area, chain_code