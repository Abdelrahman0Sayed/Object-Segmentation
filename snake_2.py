import cv2
import numpy as np
from pubsub import pub
import concurrent.futures
import asyncio
import active_contour  # Import the Cython module


class ImageProcessor:
    """Base class for image processing operations"""
    def __init__(self):
        self.img_original = None
        self.img_grayscale = None
        
    def load_image(self, file_path):
        """Load an image from file"""
        self.img_original = cv2.imread(file_path)
        if self.img_original is not None:
            self.img_grayscale = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            return True
        return False
    
    def to_grayscale(self, image):
        """Convert an image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def get_image_dimensions(self):
        """Return image dimensions if an image is loaded"""
        if self.img_original is not None:
            h, w = self.img_original.shape[:2]
            return w, h
        return 0, 0


class SnakeContour(ImageProcessor):
    """Implementation of the active contour algorithm (snake) using Cython optimizations"""
    def __init__(self):
        super().__init__()
        # Default parameters
        self.center = (0, 0)
        self.radius = 100
        self.iterations = 100
        self.alpha = 9  # Internal energy weight
        self.beta = 9   # External energy weight
        self.gamma = 1  # Balloon energy weight
        self.gradient_x = None
        self.gradient_y = None
        self._bind_events()

    def _bind_events(self):
        # pub.subscribe(self.process_snake, "snake.applied")
        pub.subscribe(self.set_initial_contour,"snake.initial")
        pub.subscribe(self.set_contour, "snake.setContour")
        pub.subscribe(self.handel_set_result, "snake.apply")

    def set_contour(self, center, radius):
        """Set new contour parameters and render initial contour"""
        self.center = center
        self.radius = radius
        image = self.render_initial_contour()
        pub.sendMessage("image.initialContour", image_contour=image)

    def set_result(self,
                   iterations,
                   alpha,
                   beta,
                   gamma,
                   center,
                   radius):
        """Apply snake algorithm with specified parameters using Cython"""
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.center = center
        self.radius = radius
        snake_points, perimeter, area, chain_code = self.process_contour_cython()
        image = self.render_result(snake_points)
        pub.sendMessage("image.snakeResult", 
                       image_result=image, 
                       perimeter=perimeter, 
                       area=area, 
                       chain_code=chain_code)

    def handel_set_result(self, iterations, alpha, beta, gamma, center, radius):
        """Handle UI request to process snake contour in a separate thread"""
        pub.sendMessage("ui.logging", message="Processing snake contour using Cython acceleration...")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, self.set_result, iterations, alpha, beta, gamma, center, radius)

    def set_initial_contour(self, image):
        """Set initial image and contour center"""
        self.img_original = image
        self.img_grayscale = self.to_grayscale(image)
        self.center = self.get_image_center()
        
        image = self.render_initial_contour()
        pub.sendMessage("image.initialContour", image_contour=image)

    def create_initial_contour(self, center=None, radius=None):
        """Generate initial circular contour points using Cython module"""
        if center is not None:
            self.center = center
        if radius is not None:
            self.radius = radius
            
        # Use Cython implementation
        return active_contour.cy_initial_contour(self.center, self.radius)

    def process_contour_cython(self):
        """Process contour using optimized Cython implementation"""
        if self.img_original is None:
            return None, 0, 0, []
        
        # Create initial contour
        snake_points = self.create_initial_contour()
        
        # Prepare image for processing
        gray_img = self.img_grayscale.copy()
        gray_img = cv2.blur(gray_img, (5, 5))  # Smooth image
        
        # Pre-compute gradient images for faster external energy calculation
        self.gradient_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        self.gradient_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

        # Use Cython implementation for contour evolution
        snake_points = active_contour.cy_evolve_contour(
            snake_points,
            gray_img,
            self.gradient_x,
            self.gradient_y,
            self.iterations,
            self.alpha,
            self.beta,
            self.gamma
        )
        
        # Calculate metrics using Cython
        perimeter, area = active_contour.cy_calculate_metrics(snake_points)
        
        # Calculate chain code using Cython
        chain_code = active_contour.cy_get_chain_code(snake_points)

        # Clean up gradients to free memory
        self.gradient_x = None
        self.gradient_y = None

        return snake_points, perimeter, area, chain_code

    # Replace the old process_contour method
    def process_contour(self):
        """Forward to Cython implementation for backward compatibility"""
        return self.process_contour_cython()

    def render_initial_contour(self):
        """Render an image with the initial contour"""
        if self.img_original is None:
            return None
            
        output_img = self.img_original.copy()
        cv2.circle(output_img, self.center, self.radius, (0, 255, 255), 2)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGBA)
        return output_img

    def get_image_center(self):
        """Return the center of the loaded image"""
        if self.img_original is not None:
            h, w = self.img_original.shape[:2]
            return w // 2, h // 2
        return 0, 0

    def render_result(self, snake_points):
        """Render the final contour on the image"""
        if self.img_original is None or snake_points is None:
            return None
            
        output_img = self.img_original.copy()
        
        # Draw initial circle for reference
        cv2.circle(output_img, self.center, self.radius, (0, 255, 255), 2)
        
        # Draw snake points
        for i in range(len(snake_points)):
            # Draw current point
            cv2.circle(output_img, snake_points[i], 4, (0, 0, 255), thickness=1)
            
            # Draw line to next point
            next_i = (i + 1) % len(snake_points)
            cv2.line(output_img, snake_points[i], snake_points[next_i], (0, 0, 255), 1)

        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGBA)
        return output_img

    # Allow using the old function names for backward compatibility
    def get_chain_code(self, snake_points):
        """Calculate chain code using Cython implementation"""
        return active_contour.cy_get_chain_code(snake_points)
