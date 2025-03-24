import cv2
import numpy as np
from pubsub import pub
import logging

class HoughTransform:
    def __init__(self):
        self.img_original = None
        self.threshold = 50
        self.shape_type = "Line"
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.blur_size = 5
        
        # Line detection parameters
        self.line_min_length = 30
        self.line_max_gap = 10
        self.line_rho = 1
        self.line_theta = np.pi/180
        
        # Circle detection parameters
        self.circle_min_radius = 10
        self.circle_max_radius = 100
        self.circle_min_dist = 20
        self.circle_param1 = 50  # higher threshold for canny
        self.circle_param2 = 30  # accumulator threshold
        
        # Ellipse detection parameters
        self.ellipse_min_area = 100
        self.ellipse_max_area = 10000
        self.ellipse_min_points = 5
        
        # Subscribe to events
        pub.subscribe(self.load_image, "hough.loadImage")
        pub.subscribe(self.process_shape, "hough.apply")
        pub.subscribe(self.update_params, "hough.updateParams")
        
    def load_image(self, image):
        """Load image for processing"""
        if image is None:
            logging.error("Received None image for Hough Transform")
            return
        
        self.img_original = image
        logging.info(f"Loaded image for Hough Transform, shape: {image.shape}")
        
    def update_params(self, param_name, value):
        """Update processing parameters"""
        if hasattr(self, param_name):
            setattr(self, param_name, value)
            logging.info(f"Updated parameter {param_name} to {value}")
        else:
            logging.error(f"Invalid parameter name: {param_name}")
            
    def process_shape(self, shape_type, threshold, **kwargs):
        """Process image based on shape type and parameters"""
        if self.img_original is None:
            logging.error("No image loaded for Hough Transform processing")
            pub.sendMessage("ui.logging", message="Error: No image loaded")
            return
            
        self.shape_type = shape_type
        self.threshold = threshold
        
        # Log the received parameters
        logging.debug(f"Received parameters: shape_type={shape_type}, threshold={threshold}, kwargs={kwargs}")
        
        # Process additional parameters
        for param_name, value in kwargs.items():
            if hasattr(self, param_name):
                setattr(self, param_name, value)
                logging.debug(f"Updated parameter {param_name} to {value}")
            else:
                logging.warning(f"Ignoring unknown parameter: {param_name}")
        
        logging.info(f"Processing Hough Transform with shape: {shape_type}, threshold: {threshold}")
        pub.sendMessage("ui.logging", message=f"Processing {shape_type} detection...")
        
        # Convert to grayscale if needed
        if self.img_original.ndim == 3:
            gray_image = cv2.cvtColor(self.img_original[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.img_original
            
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (self.blur_size, self.blur_size), 0)
            
        # Apply edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Process based on shape type
        if shape_type == "Line":
            result_image = self.detect_lines(edges, threshold)
        elif shape_type == "Circle":
            result_image = self.detect_circles(blurred, threshold)  # Use blurred for HoughCircles
        elif shape_type == "Ellipse":
            result_image = self.detect_ellipses(edges, threshold)
        else:
            logging.error(f"Invalid shape type: {shape_type}")
            pub.sendMessage("ui.logging", message="Error: Invalid shape type")
            return
            
        # Convert result to RGBA if needed
        if result_image.shape[2] == 3:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGBA)
            
        # Send results back to UI
        pub.sendMessage("hough.result", image_result=result_image)
        pub.sendMessage("ui.logging", message=f"{shape_type} detection complete")
    
    def detect_lines(self, edges, threshold):
        """Detect and draw lines using Hough Transform"""
        output_image = self.img_original.copy()
        
        # Using OpenCV's HoughLinesP for better performance and control
        lines = cv2.HoughLinesP(
            edges, 
            rho=self.line_rho,
            theta=self.line_theta, 
            threshold=threshold,
            minLineLength=self.line_min_length,
            maxLineGap=self.line_max_gap
        )
        
        line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
                line_count += 1
        
        logging.info(f"Detected {line_count} lines")
        return output_image
        
    def detect_circles(self, gray_image, threshold):
        """Detect and draw circles using Hough Transform"""
        output_image = self.img_original.copy()
        
        # Using OpenCV's HoughCircles for better control
        # Ensure param2 is positive
        param2_value = max(1, threshold)  # Ensure a minimum value of 1
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.circle_min_dist,
            param1=self.circle_param1,
            param2=param2_value,  # Use threshold with minimum value check
            minRadius=self.circle_min_radius,
            maxRadius=self.circle_max_radius
        )
        
        circle_count = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                # Draw the outer circle
                cv2.circle(output_image, center, radius, (0, 255, 0), 2, cv2.LINE_AA)
                # Draw the center of the circle
                cv2.circle(output_image, center, 2, (0, 0, 255), 3, cv2.LINE_AA)
                circle_count += 1
        
        logging.info(f"Detected {circle_count} circles")
        return output_image
        
    def detect_ellipses(self, edges, threshold):
        """Detect and draw ellipses using Hough Transform"""
        output_image = self.img_original.copy()
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        ellipse_count = 0
        for contour in contours:
            # Check if contour has enough points to fit ellipse
            if len(contour) >= self.ellipse_min_points:
                # Calculate contour area for filtering
                area = cv2.contourArea(contour)
                
                # Apply area thresholds
                if self.ellipse_min_area <= area <= self.ellipse_max_area:
                    try:
                        # Try to fit ellipse to the contour
                        ellipse = cv2.fitEllipse(contour)
                        
                        # Calculate ellipse parameters for filtering
                        center, axes, angle = ellipse
                        major_axis = max(axes) / 2
                        minor_axis = min(axes) / 2
                        
                        # Filter based on axis ratio and threshold
                        if minor_axis > 0 and (major_axis / minor_axis) < 5 and area > threshold:
                            cv2.ellipse(output_image, ellipse, (0, 255, 0), 2, cv2.LINE_AA)
                            # Draw center point
                            cv2.circle(output_image, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
                            ellipse_count += 1
                    except:
                        # Sometimes fitEllipse can fail even with enough points
                        pass
        
        logging.info(f"Detected {ellipse_count} ellipses")
        return output_image