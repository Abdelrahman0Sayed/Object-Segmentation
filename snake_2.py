import numpy as np
import cv2
import matplotlib.pyplot as plt
from pubsub import pub
import os
from datetime import datetime

# Import optimized Cython functions if available
try:
    from active_contour import (
        cy_initial_contour, cy_calc_internal_energy, 
        cy_calc_external_energy, cy_calc_balloon_energy,
        cy_contour_iteration, cy_calculate_perimeter,
        cy_calculate_polygon_area, cy_get_chain_code
    )
    USE_CYTHON = True
    print("Using Cython-optimized snake algorithm")
except ImportError:
    USE_CYTHON = False
    print("Using Python snake algorithm (Cython optimization not available)")

class SnakeProcessor:
    """
    Class to handle active contour (snake) algorithm using pubsub pattern
    for communication with UI or other components.
    """
    def __init__(self):
        # Initialize variables
        self.img_original = None
        self.img_copy = None
        self.center = (400, 350)
        self.radius = 100
        self.numOfIterations = 100
        self.alpha = 9
        self.beta = 9
        self.gamma = 1
        self.gradient_x = None
        self.gradient_y = None
        
        # Subscribe to messages
        pub.subscribe(self.on_initial_image, "snake.initial")
        pub.subscribe(self.on_set_contour, "snake.setContour")
        pub.subscribe(self.on_apply_snake, "snake.apply")
        
        self.log("Snake processor initialized")
    
    def log(self, message):
        """Send log message to UI"""
        pub.sendMessage("ui.logging", message=message)
    
    def on_initial_image(self, image):
        """Process the initial image and prepare for contour"""
        self.log("Processing initial image")
        self.img_original = image
        if self.img_original is not None:
            self.img_copy = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            # Create and publish initial contour image
            initial_contour_img = self.display_initial_contour()
            pub.sendMessage("image.initialContour", image_contour=initial_contour_img)
            self.log(f"Initial contour set: Center={self.center}, Radius={self.radius}")
        else:
            self.log("Error: Invalid image")
    
    def on_set_contour(self, center, radius):
        """Update contour parameters and display"""
        self.center = center
        self.radius = radius
        self.log(f"Contour updated: Center={center}, Radius={radius}")
        
        if self.img_original is not None:
            # Create and publish updated initial contour
            initial_contour_img = self.display_initial_contour()
            pub.sendMessage("image.initialContour", image_contour=initial_contour_img)
    
    def on_apply_snake(self, iterations, alpha, beta, gamma, center, radius):
        """Apply snake algorithm with the given parameters"""
        self.log("Applying snake algorithm...")
        
        # Update parameters
        self.numOfIterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.center = center
        self.radius = radius
        
        if self.img_original is None:
            self.log("Error: No image loaded")
            return
        
        # Run the snake algorithm
        snake_points, perimeter, area = self.contourUpdating()
        
        # Get chain code (use optimized version if available)
        if USE_CYTHON:
            chain_code = cy_get_chain_code(snake_points)
        else:
            chain_code = self.get_chain_code(snake_points)
        
        # Create output image
        result_img = self.display_output_contour(snake_points)
        
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGBA)
        # Create visualization
        self.visualize_results(snake_points, chain_code, perimeter, area)
        
        # Publish results
        pub.sendMessage("image.snakeResult", 
                        image_result=result_img,
                        perimeter=perimeter,
                        area=area,
                        chain_code=chain_code)
        
        # Also send visualization if needed
        # Show visualization directly using matplotlib instead of sending to UI
        # plt.figure(figsize=(5, 5))
        # plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        
        self.log("Snake algorithm completed")
        self.log(f"Results: Perimeter={perimeter:.2f}, Area={area:.2f}")
    
    def initial_contour(self, center, radius):
        """Generate an initial circular contour given a center and radius."""
        if USE_CYTHON:
            return cy_initial_contour(center, radius)
        
        initial_snake = []
        current_angle = 0
        resolution = 360 / 1000.0
        
        for i in range(1000):
            angle = np.array([current_angle], dtype=np.float64)
            x, y = cv2.polarToCart(
                np.array([radius], dtype=np.float64), angle, True)
            
            y_point = int(y[0][0] + center[1])
            x_point = int(x[0][0] + center[0])
            
            current_angle += resolution
            initial_snake.append((x_point, y_point))
            
        return initial_snake

    def calcInternalEnergy(self, pt, prevPt, nextPt, alpha, beta):
        """Calculate internal energy (elasticity and stiffness)"""
        if USE_CYTHON:
            return cy_calc_internal_energy(pt, prevPt, nextPt, alpha, beta)
            
        # Elasticity term (first derivative approximation)
        tension = alpha * ((pt[0] - prevPt[0])**2 + (pt[1] - prevPt[1])**2)
        
        # Stiffness term (second derivative approximation)
        curvature = beta * ((nextPt[0] - 2 * pt[0] + prevPt[0])**2 + 
                            (nextPt[1] - 2 * pt[1] + prevPt[1])**2)
        
        return tension + curvature

    def calcExternalEnergy(self, img, pt):
        """Calculate external energy based on image gradients"""
        if USE_CYTHON and self.gradient_x is not None and self.gradient_y is not None:
            return cy_calc_external_energy(self.gradient_x, self.gradient_y, pt, self.beta)
            
        h, w = img.shape[:2]
        x, y = pt
        
        # Ensure the point is within image boundaries
        if 0 <= x < w and 0 <= y < h:
            # Use pre-computed gradients if available
            if self.gradient_x is not None and self.gradient_y is not None:
                # Compute external energy as the squared gradient magnitude
                external_energy = -(self.gradient_x[y, x] ** 2 + self.gradient_y[y, x] ** 2)
                return external_energy * self.beta
            else:
                # Fallback if gradients are not pre-computed
                gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                return -(gradient_x[y, x] ** 2 + gradient_y[y, x] ** 2)
        else:
            return 0  # Return 0 for out-of-bounds points

    def calcBalloonEnergy(self, pt, prevPt, gamma):
        """Calculate balloon energy to control expansion/contraction"""
        if USE_CYTHON:
            return cy_calc_balloon_energy(pt, prevPt, gamma)
            
        dx = pt[0] - prevPt[0]
        dy = pt[1] - prevPt[1]
        return gamma * (dx*dx + dy*dy)

    def contourUpdating(self):
        """Main snake algorithm implementation"""
        snake_points = self.initial_contour(self.center, self.radius)
        grayImg = self.img_copy
        
        # Blur image to reduce noise
        grayImg = cv2.blur(grayImg, (5, 5))
        
        # Pre-compute gradient images for efficiency
        self.gradient_x = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=3)
        self.gradient_y = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=3)
        
        # Status reporting
        total_iterations = self.numOfIterations
        h, w = grayImg.shape[:2]
        
        for iteration in range(self.numOfIterations):
            if iteration % 10 == 0:
                # Log progress every 10 iterations
                progress = (iteration / total_iterations) * 100
                self.log(f"Snake progress: {progress:.1f}%")
            
            # Use optimized Cython function for the iteration if available
            if USE_CYTHON:
                snake_points = cy_contour_iteration(
                    snake_points, 
                    self.gradient_x, 
                    self.gradient_y,
                    h, w,
                    self.alpha, 
                    self.beta, 
                    self.gamma
                )
            else:
                numPoints = len(snake_points)
                newCurve = [None] * numPoints
                
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
                            if not (0 <= movePt[0] < w and 0 <= movePt[1] < h):
                                continue
                                
                            # Calculate energies
                            internal_e = self.calcInternalEnergy(
                                movePt, prevPt, nextPt, self.alpha, self.beta)
                            external_e = self.calcExternalEnergy(
                                grayImg, movePt)
                            balloon_e = self.calcBalloonEnergy(
                                movePt, prevPt, self.gamma)
                            energy = internal_e + external_e + balloon_e
                            
                            # Update if new point has lower energy
                            if energy < minEnergy:
                                minEnergy = energy
                                newPt = movePt
                    
                    newCurve[i] = newPt
                
                snake_points = newCurve
        
        # Calculate perimeter and area using optimized functions if available
        if USE_CYTHON:
            perimeter = cy_calculate_perimeter(snake_points)
            area = cy_calculate_polygon_area(snake_points)
        else:
            # Calculate perimeter
            perimeter = 0
            prevDir = 0
            for currPt, prevPt in zip(snake_points, snake_points[:-1] + [snake_points[0]]):
                dx = currPt[0] - prevPt[0]
                dy = currPt[1] - prevPt[1]
                
                # Map directions to integer codes
                dir = 0
                if dx == 0 and dy == 1:
                    dir = 0
                elif dx == -1 and dy == 1:
                    dir = 1
                elif dx == -1 and dy == 0:
                    dir = 2
                elif dx == -1 and dy == -1:
                    dir = 3
                elif dx == 0 and dy == -1:
                    dir = 4
                elif dx == 1 and dy == -1:
                    dir = 5
                elif dx == 1 and dy == 0:
                    dir = 6
                elif dx == 1 and dy == 1:
                    dir = 7
                    
                dir = (dir - prevDir + 8) % 8
                perimeter += np.sqrt(dx**2 + dy**2)  # Euclidean distance
                prevDir = dir
                
            # Calculate area
            area = self.calculate_polygon_area(snake_points)
        
        # Clean up gradients to free memory
        self.gradient_x = None
        self.gradient_y = None
        
        return snake_points, perimeter, area

    def calculate_polygon_area(self, points):
        """Compute the area of a polygon using the Shoelace Theorem."""
        n = len(points)
        area = 0
        
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            
            area += (x1 * y2) - (x2 * y1)
            
        return abs(area) / 2

    def display_output_contour(self, snake_contour):
        """Draw the initial and final snake contours on the image"""
        output_img = self.img_original.copy() if self.img_original is not None else np.zeros((700, 800, 3), dtype=np.uint8)
        
        # Draw initial circle
        cv2.circle(output_img, self.center, self.radius, (0, 255, 255), 2)
        
        # Draw snake points and connections
        for i in range(len(snake_contour)):
            # Draw point
            cv2.circle(output_img, snake_contour[i], 4, (0, 0, 255), thickness=1)
            
            # Draw line to next point
            next_i = (i + 1) % len(snake_contour)
            cv2.line(output_img, snake_contour[i], snake_contour[next_i], (0, 0, 255), 1)
            
        return output_img

    def display_initial_contour(self):
        """Create image with initial contour only"""
        output_img = self.img_original.copy() if self.img_original is not None else np.zeros((700, 800, 3), dtype=np.uint8)
        cv2.circle(output_img, self.center, self.radius, (0, 255, 255), 2)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGBA)
        return output_img

    def get_chain_code(self, snake_points):
        """Calculate the chain code representation of the contour"""
        chain_code = []
        for i in range(len(snake_points)):
            current = snake_points[i]
            next_point = snake_points[(i + 1) % len(snake_points)]
            
            # Calculate direction vector
            dx = next_point[0] - current[0]
            dy = next_point[1] - current[1]
            
            # Normalize direction
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
                
            # Convert to chain code (8-direction)
            if dx == 0 and dy == -1:    # North
                code = 0
            elif dx == 1 and dy == -1:   # Northeast
                code = 1
            elif dx == 1 and dy == 0:    # East
                code = 2
            elif dx == 1 and dy == 1:    # Southeast
                code = 3
            elif dx == 0 and dy == 1:    # South
                code = 4
            elif dx == -1 and dy == 1:   # Southwest
                code = 5
            elif dx == -1 and dy == 0:   # West
                code = 6
            elif dx == -1 and dy == -1:  # Northwest
                code = 7
            else:
                code = -1  # Invalid direction
                
            chain_code.append(code)
        
        return chain_code

    def visualize_results(self, snake_points, chain_code, perimeter, area):
        """
        Create a simple visualization showing only the key metrics
        """
        plt.figure(figsize=(4, 3))
        plt.axis('off')
        
        # Format chain code representation
        if len(chain_code) > 20:
            chain_code_str = (
                f"First 10: {' '.join(str(c) for c in chain_code[:10])}\n"
                f"Last 10: {' '.join(str(c) for c in chain_code[-10:])}"
            )
        else:
            chain_code_str = f"All: {' '.join(str(c) for c in chain_code)}"
        
        # Create metrics text with clear explanations
        metrics_text = (
            f"SNAKE CONTOUR RESULTS\n\n"
            f"Perimeter: {perimeter:.2f} pixels\n"
            f"Area: {area:.2f} square pixels\n"
            f"Number of points: {len(snake_points)}\n\n"
            f"Chain Code ({len(chain_code)} directions):\n{chain_code_str}\n\n"
            f"* Chain code represents movement direction between points\n"
            f"* 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW"
        )
        
        plt.text(0.5, 0.5, metrics_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                 fontsize=12)
        
        plt.tight_layout()
        plt.show()
