# main.py
import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QFileDialog, QSlider, QSpinBox, 
                            QGroupBox, QFormLayout, QStatusBar)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint

class Image:
    """Base image class as shown in the original code"""
    def __init__(self):
        self.img_original = None
        self.img_copy = None
        
    def load_image(self, file_path):
        self.img_original = cv2.imread(file_path)
        if self.img_original is not None:
            self.img_copy = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            return True
        return False


# to implement the active contour (or “snake”) algorithm.  automatically adjust an initial contour.
#  This is done by minimizing an energy function that combines internal, external and ballon
class ActiveContour(Image):
    def __init__(self):
        super().__init__()
        self.center = (400, 350)
        self.radius = 100
        self.numOfIterations = 100
        self.alpha = 9
        self.beta = 9
        self.gamma = 1
        # Initialize gradient images as None
        self.gradient_x = None
        self.gradient_y = None

    #Generate an initial circular contour given a center and radius.
    def initial_contour(self, center, radius):
        initial_snake = []
        current_angle = 0 # the angle starts from 0
        #360 because we want to form a circle that have 360 degrees
        resolution = 360 / 1000.0
        # we have 1000 points to form the circle, each iteration we add a point to the circle
        for i in range(1000):
            # Ensure angle has dtype=np.float32
            angle = np.array([current_angle], dtype=np.float64)
            #conversion from angle and radius to x and y coordinates
            x, y = cv2.polarToCart(
                np.array([radius], dtype=np.float64), angle, True)
            # Access the first elements of x and y arrays
            # for example x looks like this array([[-100.]])
            # so we access the first element of the array which is -100
            y_point = int(y[0][0] + center[1]) 
            x_point = int(x[0][0] + center[0]) #  shifts the x-coordinate.

            current_angle += resolution # This means that with each iteration of the loop, the angle increases by 0.36 degrees. 
            initial_snake.append((x_point, y_point))
        return initial_snake

    def calcInternalEnergy(self, pt, prevPt, nextPt, alpha, beta):
        # Elasticity term (first derivative approximation)
        tension = alpha * ((pt[0] - prevPt[0])**2 + (pt[1] - prevPt[1])**2) #(x2-x1)^2 + (y2-y1)^2
        
        # Stiffness term (second derivative approximation)
        curvature = beta * ((nextPt[0] - 2 * pt[0] + prevPt[0])**2 + (nextPt[1] - 2 * pt[1] + prevPt[1])**2)# (x3-2x2+x1)^2 + (y3-2y2+y1)^2
        
        return tension + curvature

    def calcExternalEnergy(self, img, pt):
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
        dx = pt[0] - prevPt[0]
        dy = pt[1] - prevPt[1]
        return gamma * (dx*dx + dy*dy)

    def contourUpdating(self):
        snake_points = self.initial_contour(self.center, self.radius)
        grayImg = self.img_copy
        #Converts the image to grayscale and applies a blur filter to reduce noise.
        grayImg = cv2.blur(grayImg, (5, 5))
        
        # Pre-compute gradient images
        self.gradient_x = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=3)
        self.gradient_y = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=3)

        for _ in range(self.numOfIterations):
            numPoints = len(snake_points)
            newCurve = [None] * numPoints #  stores the updated contour points.

            for i in range(numPoints):
                pt = snake_points[i]
                prevPt = snake_points[(i - 1 + numPoints) % numPoints]
                nextPt = snake_points[(i + 1) % numPoints]
                minEnergy = float('inf')
                newPt = pt

                #iterates over a 3×3 neighborhood around the current point pt.
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        movePt = (pt[0] + dx, pt[1] + dy)
                        
                        # Check if the move point is within image boundaries
                        h, w = grayImg.shape[:2]
                        #if the move point is outside the image boundaries, we skip the current iteration.
                        if not (0 <= movePt[0] < w and 0 <= movePt[1] < h):
                            continue
                            
                        internal_e = self.calcInternalEnergy(
                            movePt, prevPt, nextPt, self.alpha, self.beta)
                        external_e = self.calcExternalEnergy(
                            grayImg, movePt)
                        balloon_e = self.calcBalloonEnergy(
                            movePt, prevPt, self.gamma)
                        energy = internal_e + external_e + balloon_e

                        #updates only if the new point has lower energy than the current point.
                        if energy < minEnergy:
                            minEnergy = energy
                            newPt = movePt

                newCurve[i] = newPt

            snake_points = newCurve

        perimeter = 0
        prevDir = 0
        for currPt, prevPt in zip(snake_points, snake_points[:-1] + [snake_points[0]]):
            dx = currPt[0] - prevPt[0]
            dy = currPt[1] - prevPt[1]

            dir = 0  # Initialize dir before any conditional assignment
            # Map directions to integer codes (adjust based on your chain code convention)
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

            dir = (dir - prevDir + 8) % 8  # Calculation is always performed
            perimeter += np.sqrt(dx**2 + dy**2) # Euclidean distance
            prevDir = dir

        # Calculate area using polygonal approximation (for 8-direction chain code)
        approx_polygon = np.array(snake_points, dtype=np.int32)

        area = self.calculate_polygon_area(snake_points)
        #area = cv2.contourArea(approx_polygon)
        # Clean up gradients to free memory
        self.gradient_x = None
        self.gradient_y = None

        return snake_points, perimeter, area


    def calculate_polygon_area(self, points):
        """
        Compute the area of a polygon using the Shoelace Theorem.
        
        :param points: List of (x, y) tuples representing the contour.
        :return: Area of the polygon.
        """
        n = len(points)  # Number of points in the polygon
        area = 0

        for i in range(n):
            x1, y1 = points[i]  # Current vertex
            x2, y2 = points[(i + 1) % n]  # Next vertex (wrapping around)

            area += (x1 * y2) - (x2 * y1)

        return abs(area) / 2  # Absolute value and divide by 2


    def display_output_contour(self, snake_contour, output_image):
        # Draw the initial and final snake contours on the image
        output_img = self.img_original.copy() if self.img_original is not None else np.zeros((700, 800, 3), dtype=np.uint8)
        cv2.circle(output_img, self.center, self.radius,
                   (0, 255, 255), 2)  # Draw initial circle
        for i in range(len(snake_contour)):
            cv2.circle(
                output_img, snake_contour[i], 4, (0, 0, 255), thickness=1)
            # Draw line between current and next point (cyclic connection)
            next_i = (i + 1) % len(snake_contour)
            # blue for connection
            cv2.line(output_img, snake_contour[i],
                     snake_contour[next_i], (0, 0, 255), 1)

        cv2.line(output_img, snake_contour[0],
                 snake_contour[-1], (0, 0, 255), 1)
        
        return output_img

    def display_initial_contour(self):
        output_img = self.img_original.copy() if self.img_original is not None else np.zeros((700, 800, 3), dtype=np.uint8)
        cv2.circle(output_img, self.center, self.radius,
                   (0, 255, 255), 2)
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

class ContourWorker(QThread):
    finished = pyqtSignal(object, float, float, list)  # Added chain_code to signal
    progress = pyqtSignal(int)
    
    def __init__(self, active_contour):
        super().__init__()
        self.active_contour = active_contour
        
    def run(self):
        snake_points, perimeter, area = self.active_contour.contourUpdating()
        chain_code = self.active_contour.get_chain_code(snake_points)
        self.finished.emit(snake_points, perimeter, area, chain_code)


class ImageViewer(QLabel):
    contourChanged = pyqtSignal(tuple, int)  # Signal to emit center and radius changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.setText("No Image")
        
        # For drag handling
        self.is_dragging = False
        self.drag_start = None
        self.center = (0, 0)
        self.radius = 0
        self.edge_drag = False  # Flag to determine if dragging the edge (radius adjustment)
        self.edge_threshold = 10  # Pixels threshold to detect edge
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Original image and display scale factors
        self.original_image = None
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        self.display_size = None
        
    def display_image(self, cv_img, auto_resize=True):
        if cv_img is None:
            self.setText("No Image")
            self.original_image = None
            return
            
        # Store original image
        self.original_image = cv_img.copy()
        
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        
        # Convert the image to RGB format (OpenCV uses BGR)
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        q_img = QImage(cv_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_img)
        
        if auto_resize:
            self.display_size = (self.width(), self.height())
            pixmap = pixmap.scaled(self.width(), self.height(), 
                                  Qt.AspectRatioMode.KeepAspectRatio)
            
            # Calculate scale factors
            scaled_w = pixmap.width()
            scaled_h = pixmap.height()
            self.scale_factor_x = w / scaled_w
            self.scale_factor_y = h / scaled_h
        else:
            self.scale_factor_x = 1.0
            self.scale_factor_y = 1.0
            
        self.setPixmap(pixmap)
        self.setText("")  # Clear text when image is displayed
    
    def set_contour_params(self, center, radius):
        """Set the current contour parameters"""
        self.center = center
        self.radius = radius
    
    def mousePressEvent(self, event):
        if self.original_image is None:
            return
        
        # Get position in image coordinates
        pos = self.get_image_coordinates(event.position())
        
        # Check if click is near the circle edge
        distance = self.distance_to_center(pos)
        if abs(distance - self.radius) < self.edge_threshold:
            self.is_dragging = True
            self.edge_drag = True
            self.drag_start = pos
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        # Check if click is inside the circle
        elif distance < self.radius:
            self.is_dragging = True
            self.edge_drag = False
            self.drag_start = pos
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
    
    def mouseMoveEvent(self, event):
        if self.original_image is None:
            return
        
        # Get position in image coordinates
        pos = self.get_image_coordinates(event.position())
        
        # For cursor changes on hover
        if not self.is_dragging:
            distance = self.distance_to_center(pos)
            if abs(distance - self.radius) < self.edge_threshold:
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            elif distance < self.radius:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            return
        
        if self.is_dragging:
            if self.edge_drag:
                # Adjust radius
                new_radius = int(self.distance_to_center(pos))
                if new_radius > 10:  # Minimum radius
                    self.radius = new_radius
                    self.contourChanged.emit(self.center, self.radius)
            else:
                # Move center
                dx = pos[0] - self.drag_start[0]
                dy = pos[1] - self.drag_start[1]
                self.center = (self.center[0] + dx, self.center[1] + dy)
                self.drag_start = pos
                self.contourChanged.emit(self.center, self.radius)
    
    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.edge_drag = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    def get_image_coordinates(self, qt_pos):
        """Convert Qt coordinate to original image coordinate"""
        # Get widget position relative to content
        widget_w = self.width()
        widget_h = self.height()
        
        if self.pixmap():
            pixmap_w = self.pixmap().width()
            pixmap_h = self.pixmap().height()
            
            # Calculate offsets if image is centered
            offset_x = (widget_w - pixmap_w) / 2
            offset_y = (widget_h - pixmap_h) / 2
            
            # Adjust position to account for centering
            pos_x = qt_pos.x() - offset_x
            pos_y = qt_pos.y() - offset_y
            
            # Convert to original image coordinates
            orig_x = int(pos_x * self.scale_factor_x)
            orig_y = int(pos_y * self.scale_factor_y)
            
            return (orig_x, orig_y)
        return (0, 0)
    
    def distance_to_center(self, pos):
        """Calculate distance from point to center"""
        return ((pos[0] - self.center[0])**2 + (pos[1] - self.center[1])**2)**0.5


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.active_contour = ActiveContour()
        self.contour_result = None
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Active Contour Segmentation")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Image loading section
        image_group = QGroupBox("Image")
        image_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_button)
        
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)
        
        # Parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        # Center X
        center_x_layout = QHBoxLayout()
        self.center_x_spin = QSpinBox()
        self.center_x_spin.setRange(0, 2000)
        self.center_x_spin.setValue(self.active_contour.center[0])
        center_x_layout.addWidget(self.center_x_spin)
        params_layout.addRow("Center X:", center_x_layout)
        
        # Center Y
        center_y_layout = QHBoxLayout()
        self.center_y_spin = QSpinBox()
        self.center_y_spin.setRange(0, 2000)
        self.center_y_spin.setValue(self.active_contour.center[1])
        center_y_layout.addWidget(self.center_y_spin)
        params_layout.addRow("Center Y:", center_y_layout)
        
        # Radius
        radius_layout = QHBoxLayout()
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(10, 500)
        self.radius_spin.setValue(self.active_contour.radius)
        radius_layout.addWidget(self.radius_spin)
        params_layout.addRow("Radius:", radius_layout)
        
        # Connect parameter changes
        self.center_x_spin.valueChanged.connect(self.update_parameters_from_controls)
        self.center_y_spin.valueChanged.connect(self.update_parameters_from_controls)
        self.radius_spin.valueChanged.connect(self.update_parameters_from_controls)
        
        # Add note about dragging
        drag_info = QLabel("Tip: Drag inside circle to move it.\nDrag edge to resize.")
        drag_info.setStyleSheet("color: #555; font-style: italic;")
        params_layout.addRow("", drag_info)
        
        # Alpha (internal energy)
        alpha_layout = QHBoxLayout()
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(0, 50)
        self.alpha_spin.setValue(self.active_contour.alpha)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 50)
        self.alpha_slider.setValue(self.active_contour.alpha)
        alpha_layout.addWidget(self.alpha_spin)
        alpha_layout.addWidget(self.alpha_slider)
        params_layout.addRow("Alpha (Internal):", alpha_layout)
        
        # Connect spinner and slider
        self.alpha_spin.valueChanged.connect(self.alpha_slider.setValue)
        self.alpha_slider.valueChanged.connect(self.alpha_spin.setValue)
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        
        # Beta (external energy)
        beta_layout = QHBoxLayout()
        self.beta_spin = QSpinBox()
        self.beta_spin.setRange(0, 50)
        self.beta_spin.setValue(self.active_contour.beta)
        self.beta_slider = QSlider(Qt.Orientation.Horizontal)
        self.beta_slider.setRange(0, 50)
        self.beta_slider.setValue(self.active_contour.beta)
        beta_layout.addWidget(self.beta_spin)
        beta_layout.addWidget(self.beta_slider)
        params_layout.addRow("Beta (External):", beta_layout)
        
        # Connect spinner and slider
        self.beta_spin.valueChanged.connect(self.beta_slider.setValue)
        self.beta_slider.valueChanged.connect(self.beta_spin.setValue)
        self.beta_slider.valueChanged.connect(self.update_beta)
        
        # Gamma (balloon energy)
        gamma_layout = QHBoxLayout()
        self.gamma_spin = QSpinBox()
        self.gamma_spin.setRange(0, 20)
        self.gamma_spin.setValue(self.active_contour.gamma)
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(0, 20)
        self.gamma_slider.setValue(self.active_contour.gamma)
        gamma_layout.addWidget(self.gamma_spin)
        gamma_layout.addWidget(self.gamma_slider)
        params_layout.addRow("Gamma (Balloon):", gamma_layout)
        
        # Connect spinner and slider
        self.gamma_spin.valueChanged.connect(self.gamma_slider.setValue)
        self.gamma_slider.valueChanged.connect(self.gamma_spin.setValue)
        self.gamma_slider.valueChanged.connect(self.update_gamma)
        
        # Iterations
        iter_layout = QHBoxLayout()
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 500)
        self.iter_spin.setValue(self.active_contour.numOfIterations)
        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(10, 500)
        self.iter_slider.setValue(self.active_contour.numOfIterations)
        iter_layout.addWidget(self.iter_spin)
        iter_layout.addWidget(self.iter_slider)
        params_layout.addRow("Iterations:", iter_layout)
        
        # Connect spinner and slider
        self.iter_spin.valueChanged.connect(self.iter_slider.setValue)
        self.iter_slider.valueChanged.connect(self.iter_spin.setValue)
        self.iter_slider.valueChanged.connect(self.update_iterations)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        self.show_initial_button = QPushButton("Show Initial Contour")
        self.show_initial_button.clicked.connect(self.show_initial_contour)
        action_layout.addWidget(self.show_initial_button)
        
        self.run_button = QPushButton("Run Active Contour")
        self.run_button.clicked.connect(self.run_active_contour)
        action_layout.addWidget(self.run_button)
        
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        action_layout.addWidget(self.save_button)
        
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        
        self.perimeter_label = QLabel("0.0")
        results_layout.addRow("Perimeter:", self.perimeter_label)
        
        self.area_label = QLabel("0.0")
        results_layout.addRow("Area:", self.area_label)
        
        self.chain_code_label = QLabel("")
        self.chain_code_label.setWordWrap(True)
        results_layout.addRow("Chain Code:", self.chain_code_label)

        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        # Set left panel layout
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Create right panel (image view)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.image_viewer = ImageViewer()
        self.image_viewer.contourChanged.connect(self.update_parameters_from_drag)
        right_layout.addWidget(self.image_viewer)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # Give more space to the image viewer
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize worker
        self.contour_worker = None
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        
        if file_path:
            if self.active_contour.load_image(file_path):
                # Display the loaded image
                self.show_initial_contour()
                self.status_bar.showMessage(f"Loaded image: {file_path}")
                
                # Update center sliders based on image dimensions
                if self.active_contour.img_original is not None:
                    h, w = self.active_contour.img_original.shape[:2]
                    self.center_x_spin.setMaximum(w)
                    self.center_y_spin.setMaximum(h)
                    
                    # Set default center to image center
                    self.center_x_spin.setValue(w // 2)
                    self.center_y_spin.setValue(h // 2)
                    self.active_contour.center = (w // 2, h // 2)
                    
                    # Update viewer with new contour params
                    self.image_viewer.set_contour_params(self.active_contour.center, self.active_contour.radius)
                    self.show_initial_contour()
            else:
                self.status_bar.showMessage("Failed to load image")
    
    def update_parameters_from_controls(self):
        """Update parameters when controls are changed"""
        x = self.center_x_spin.value()
        y = self.center_y_spin.value()
        radius = self.radius_spin.value()
        
        self.active_contour.center = (x, y)
        self.active_contour.radius = radius
        
        # Update viewer
        self.image_viewer.set_contour_params(self.active_contour.center, self.active_contour.radius)
        self.show_initial_contour()
    
    def update_parameters_from_drag(self, center, radius):
        """Update parameters when dragging on the image"""
        self.active_contour.center = center
        self.active_contour.radius = radius
        
        # Update controls
        self.center_x_spin.setValue(center[0])
        self.center_y_spin.setValue(center[1])
        self.radius_spin.setValue(radius)
        
        # Refresh display
        self.show_initial_contour()
    
    def update_alpha(self):
        self.active_contour.alpha = self.alpha_spin.value()
    
    def update_beta(self):
        self.active_contour.beta = self.beta_spin.value()
    
    def update_gamma(self):
        self.active_contour.gamma = self.gamma_spin.value()
    
    def update_iterations(self):
        self.active_contour.numOfIterations = self.iter_spin.value()
    
    def show_initial_contour(self):
        if self.active_contour.img_original is None:
            self.status_bar.showMessage("No image loaded")
            return
            
        initial_img = self.active_contour.display_initial_contour()
        self.image_viewer.display_image(initial_img)
        self.image_viewer.set_contour_params(self.active_contour.center, self.active_contour.radius)
        self.status_bar.showMessage(f"Initial contour: Center={self.active_contour.center}, Radius={self.active_contour.radius}")
    
    def run_active_contour(self):
        if self.active_contour.img_original is None:
            self.status_bar.showMessage("No image loaded")
            return
            
        self.status_bar.showMessage("Running active contour algorithm...")
        self.run_button.setEnabled(False)
        
        # Create and start worker thread
        self.contour_worker = ContourWorker(self.active_contour)
        self.contour_worker.finished.connect(self.on_contour_finished)
        self.contour_worker.start()
    
    def on_contour_finished(self, snake_points, perimeter, area, chain_code):
        self.contour_result = snake_points
        
        # Display results
        result_img = self.active_contour.display_output_contour(snake_points, None)
        self.image_viewer.display_image(result_img)
        
        # Update measurements
        self.perimeter_label.setText(f"{perimeter:.2f}")
        self.area_label.setText(f"{area:.2f}")
        
        # Display chain code with better formatting
        chain_code_str = ' '.join(map(str, chain_code))
        if len(chain_code_str) > 100:
            # Show first 50 and last 50 characters with counter in middle
            first_part = ' '.join(map(str, chain_code[:25]))
            last_part = ' '.join(map(str, chain_code[-25:]))
            total_codes = len(chain_code)
            chain_code_str = f"{first_part} ... [{total_codes} codes] ... {last_part}"
        self.chain_code_label.setText(chain_code_str)
        
        self.status_bar.showMessage("Active contour completed")
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
    
    def save_result(self):
        if self.contour_result is None or self.active_contour.img_original is None:
            self.status_bar.showMessage("No result to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            result_img = self.active_contour.display_output_contour(self.contour_result, None)
            cv2.imwrite(file_path, result_img)
            self.status_bar.showMessage(f"Result saved to: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())