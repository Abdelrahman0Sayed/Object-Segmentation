import cv2
import numpy as np
from pubsub import pub
import concurrent.futures
import asyncio


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
    """Implementation of the active contour algorithm (snake)"""
    def __init__(self):
        super().__init__()
        # Default parameters
        self.center = (0, 0)
        self.radius = 100
        self.iterations = 100
        self.alpha = 9  # Internal energy weight
        self.beta = 9   # External energy weight
        self.gamma = 1  # Balloon energy weight
        self._bind_events()

    def _bind_events(self):
        # pub.subscribe(self.process_snake, "snake.applied")
        pub.subscribe(self.set_initial_contour,"snake.initial")
        pub.subscribe(self.set_contour, "snake.setContour")
        pub.subscribe(self.handel_set_result, "snake.apply")

    def set_contour(self, center, radius):
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
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.center = center
        self.radius = radius
        snake_points, perimeter, area, chain_code = self.process_contour()
        image = self.render_result(snake_points)
        pub.sendMessage("image.snakeResult", image_result=image, perimeter=perimeter, area=area, chain_code=chain_code)

    def handel_set_result(self, iterations, alpha, beta, gamma, center, radius):
        pub.sendMessage("ui.logging", message="Processing snake contour...")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, self.set_result, iterations, alpha, beta, gamma, center, radius)

    def set_initial_contour(self,image):
        self.img_original = image
        self.img_grayscale = self.to_grayscale(image)
        self.center = self.get_image_center()
        
        image = self.render_initial_contour()
        pub.sendMessage("image.initialContour", image_contour=image)

    def create_initial_contour(self, center=None, radius=None):
        """Generate initial circular contour points"""
        if center is not None:
            self.center = center
        if radius is not None:
            self.radius = radius
            
        initial_snake = []
        current_angle = 0
        resolution = 360 / 1000.0  # 1000 points around circle
        
        for i in range(1000):
            angle = np.array([current_angle], dtype=np.float64)
            x, y = cv2.polarToCart(
                np.array([self.radius], dtype=np.float64), 
                angle, 
                True
            )

            y_point = int(y[0][0] + self.center[1])
            x_point = int(x[0][0] + self.center[0])
            
            current_angle += resolution
            initial_snake.append((x_point, y_point))
            
        return initial_snake

    def _calculate_internal_energy(self, pt, prev_pt, next_pt, alpha):
        """Calculate internal energy based on curvature"""
        dx1 = pt[0] - prev_pt[0]
        dy1 = pt[1] - prev_pt[1]
        dx2 = next_pt[0] - pt[0]
        dy2 = next_pt[1] - pt[1]

        denominator = pow(dx1*dx1 + dy1*dy1, 1.5)
        if denominator == 0:
            return 0  # Handle division by zero

        curvature = (dx1 * dy2 - dx2 * dy1) / denominator
        return alpha * curvature

    def _calculate_external_energy(self, img, pt, beta):
        """Calculate external energy based on image gradient"""
        h, w = img.shape[:2]
        x, y = pt
        if 0 <= x < w and 0 <= y < h:
            # Higher intensity for stronger edges
            return -beta * img[y, x]
        else:
            return 0  # Out-of-bounds points

    def _calculate_balloon_energy(self, pt, prev_pt, gamma):
        """Calculate balloon energy to inflate/deflate contour"""
        dx = pt[0] - prev_pt[0]
        dy = pt[1] - prev_pt[1]
        return gamma * (dx*dx + dy*dy)

    def process_contour(self):
        """Main contour processing function"""
        if self.img_original is None:
            return None, 0, 0, []
        
        # Create initial contour
        snake_points = self.create_initial_contour()
        
        # Prepare image for processing
        gray_img = self.img_grayscale.copy()
        gray_img = cv2.blur(gray_img, (5, 5))  # Smooth image

        # Evolve contour through iterations
        for _ in range(self.iterations):
            num_points = len(snake_points)
            new_curve = [None] * num_points

            for i in range(num_points):
                pt = snake_points[i]
                prev_pt = snake_points[(i - 1 + num_points) % num_points]
                next_pt = snake_points[(i + 1) % num_points]
                min_energy = float('inf')
                new_pt = pt

                # Check all neighboring pixels
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        move_pt = (pt[0] + dx, pt[1] + dy)
                        
                        # Check boundaries
                        h, w = gray_img.shape[:2]
                        if not (0 <= move_pt[0] < w and 0 <= move_pt[1] < h):
                            continue
                            
                        # Calculate energies
                        internal_e = self._calculate_internal_energy(
                            move_pt, prev_pt, next_pt, self.alpha)
                        external_e = self._calculate_external_energy(
                            gray_img, move_pt, self.beta)
                        balloon_e = self._calculate_balloon_energy(
                            move_pt, prev_pt, self.gamma)
                        
                        # Total energy
                        energy = internal_e + external_e + balloon_e

                        # Keep minimum energy point
                        if energy < min_energy:
                            min_energy = energy
                            new_pt = move_pt

                new_curve[i] = new_pt

            snake_points = new_curve

        # Calculate perimeter
        perimeter = 0
        for i in range(len(snake_points)):
            curr = snake_points[i]
            next_pt = snake_points[(i + 1) % len(snake_points)]
            dx = curr[0] - next_pt[0]
            dy = curr[1] - next_pt[1]
            perimeter += np.sqrt(dx**2 + dy**2)

        # Calculate area
        contour_array = np.array(snake_points, dtype=np.int32)
        area = cv2.contourArea(contour_array)

        # Calculate chain code
        chain_code = self.get_chain_code(snake_points)

        return snake_points, perimeter, area, chain_code

    def get_chain_code(self, snake_points):
        """Calculate the 8-direction chain code of contour"""
        chain_code = []
        for i in range(len(snake_points)):
            current = snake_points[i]
            next_point = snake_points[(i + 1) % len(snake_points)]
            
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
