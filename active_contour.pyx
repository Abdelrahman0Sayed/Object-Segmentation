# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, cos, sin, M_PI

ctypedef np.float64_t DTYPE_t

def cy_initial_contour(tuple center, int radius, int num_points=1000):
    """Optimized initial contour generation using Cython"""
    cdef:
        list initial_snake = []
        double current_angle = 0.0
        double resolution = 360.0 / num_points
        double angle, x, y
        int y_point, x_point, i
        double c_x = center[0]
        double c_y = center[1]
        double cos_val, sin_val
        
    for i in range(num_points):
        angle = current_angle * M_PI / 180.0  # Convert to radians
        
        # Manual calculation of cos/sin for polarToCart
        cos_val = cos(angle)
        sin_val = sin(angle)
        
        x = radius * cos_val
        y = radius * sin_val
        
        y_point = int(y + c_y)
        x_point = int(x + c_x)
        
        current_angle += resolution
        initial_snake.append((x_point, y_point))
        
    return initial_snake

def cy_calc_internal_energy(tuple pt, tuple prevPt, tuple nextPt, double alpha, double beta):
    """Optimized internal energy calculation using Cython"""
    cdef:
        double tension, curvature
        double dx1 = pt[0] - prevPt[0]
        double dy1 = pt[1] - prevPt[1]
        double dx2 = nextPt[0] - 2 * pt[0] + prevPt[0]
        double dy2 = nextPt[1] - 2 * pt[1] + prevPt[1]
    
    # Elasticity term (first derivative)
    tension = alpha * (dx1*dx1 + dy1*dy1)
    
    # Stiffness term (second derivative)
    curvature = beta * (dx2*dx2 + dy2*dy2)
    
    return tension + curvature

def cy_calc_external_energy(np.ndarray[DTYPE_t, ndim=2] gradient_x, 
                           np.ndarray[DTYPE_t, ndim=2] gradient_y, 
                           tuple pt, double beta):
    """Optimized external energy calculation using Cython"""
    cdef:
        int x = pt[0]
        int y = pt[1]
        int h = gradient_x.shape[0]
        int w = gradient_x.shape[1]
        double external_energy = 0.0
    
    # Ensure point is within image boundaries
    if 0 <= x < w and 0 <= y < h:
        # Compute external energy as negative squared gradient magnitude
        external_energy = -(gradient_x[y, x] * gradient_x[y, x] + 
                           gradient_y[y, x] * gradient_y[y, x])
        return external_energy * beta
    else:
        return 0.0

def cy_calc_balloon_energy(tuple pt, tuple prevPt, double gamma):
    """Optimized balloon energy calculation using Cython"""
    cdef:
        double dx = pt[0] - prevPt[0]
        double dy = pt[1] - prevPt[1]
    
    return gamma * (dx*dx + dy*dy)

def cy_contour_iteration(list snake_points, 
                        np.ndarray[DTYPE_t, ndim=2] gradient_x,
                        np.ndarray[DTYPE_t, ndim=2] gradient_y,
                        int img_height, int img_width,
                        double alpha, double beta, double gamma):
    """Optimized contour iteration using Cython - fixed to match Python implementation"""
    cdef:
        int numPoints = len(snake_points)
        list newCurve = [None] * numPoints
        int i, dx, dy
        tuple pt, prevPt, nextPt, movePt, newPt
        double minEnergy, energy, internal_e, external_e, balloon_e
    
    # Debug print for parameters
    # print(f"Cython iteration with: alpha={alpha}, beta={beta}, gamma={gamma}")
    
    for i in range(numPoints):
        pt = snake_points[i]
        prevPt = snake_points[(i - 1 + numPoints) % numPoints]
        nextPt = snake_points[(i + 1) % numPoints]
        minEnergy = float('inf')
        newPt = pt
        
        # Search in 3x3 neighborhood
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                movePt = (pt[0] + dx, pt[1] + dy)
                
                # Check if the move point is within image boundaries
                if not (0 <= movePt[0] < img_width and 0 <= movePt[1] < img_height):
                    continue
                
                # Calculate energies
                internal_e = cy_calc_internal_energy(movePt, prevPt, nextPt, alpha, beta)
                external_e = cy_calc_external_energy(gradient_x, gradient_y, movePt, beta)
                balloon_e = cy_calc_balloon_energy(movePt, prevPt, gamma)
                
                # Total energy (same formula as in Python version)
                energy = internal_e + external_e + balloon_e
                
                # Update if new point has lower energy
                if energy < minEnergy:
                    minEnergy = energy
                    newPt = movePt
        
        newCurve[i] = newPt
    
    return newCurve

def cy_calculate_perimeter(list snake_points):
    """Calculate perimeter of the contour"""
    cdef:
        int i, numPoints = len(snake_points)
        double perimeter = 0.0
        double dx, dy  # Changed to double for more precision
        int x1, y1, x2, y2
    
    if numPoints <= 1:
        return 0.0  # Handle edge case
    
    for i in range(numPoints):
        # Safely extract coordinates using direct indexing
        x1 = snake_points[i][0]
        y1 = snake_points[i][1]
        
        # Get previous point with safe wraparound
        if i == 0:
            x2 = snake_points[numPoints-1][0]
            y2 = snake_points[numPoints-1][1]
        else:
            x2 = snake_points[i-1][0]
            y2 = snake_points[i-1][1]
        
        # Calculate distance
        dx = x1 - x2
        dy = y1 - y2
        perimeter += sqrt(dx*dx + dy*dy)
    
    return perimeter

def cy_calculate_polygon_area(list points):
    """Compute the area of a polygon using the Shoelace Theorem"""
    cdef:
        int i, n = len(points)
        double area = 0.0
        int x1, y1, x2, y2
    
    if n <= 2:
        return 0.0  # A polygon needs at least 3 points to have area
    
    for i in range(n):
        # Safely extract current point coordinates
        x1 = points[i][0]
        y1 = points[i][1]
        
        # Get next point with safe wraparound
        if i == n-1:
            x2 = points[0][0]
            y2 = points[0][1]
        else:
            x2 = points[i+1][0]
            y2 = points[i+1][1]
        
        area += (x1 * y2) - (x2 * y1)
    
    return fabs(area) / 2.0

def cy_get_chain_code(list snake_points):
    """Calculate the chain code representation of the contour"""
    cdef:
        int i, numPoints = len(snake_points)
        list chain_code = []
        int dx, dy, code
        tuple current, next_point
    
    for i in range(numPoints):
        current = snake_points[i]
        next_point = snake_points[(i + 1) % numPoints]
        
        # Calculate direction vector
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        
        # Normalize direction
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        
        # Convert to chain code (8-direction)
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