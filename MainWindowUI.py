from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QTimer, QPointF
from pubsub import pub
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging
import numpy as np
import math
import cv2

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # "w" overwrites the file; use "a" to append
)


class MainWindowUI(QMainWindow):
    def __init__(self):
        super(MainWindowUI, self).__init__()
        self.ui = uic.loadUi("image-processing.ui", self)
        self.ui.show()
        self.ui.setWindowTitle("Image Filter")
        self._bind_events()
        self._bind_ui_events()
        self.image = None
        self.image_output = None
        self.dragging = False
        self.adjusting_radius = False
        self.last_pos = None
        self._setup_mouse_tracking()  # Initialize mouse tracking

    def _bind_events(self):
        pub.subscribe(self.set_snake_output, "image.snakeResult")
        pub.subscribe(self.set_initial_contour, "image.initialContour")
        pub.subscribe(self.set_hough_output, "hough.result")  # Add this line
        pub.subscribe(self.add_logging, "ui.logging")
        pub.subscribe(self.clear_logging, "ui.clearLogging")

    def _bind_ui_events(self):
        self.ui.btnLoadImage.clicked.connect(self.upload_image)
        self.ui.btnReset.clicked.connect(self.reset_image)
        self.ui.tabWidget.currentChanged.connect(self.on_tab_changed)  # Modify this line
        self.ui.spinBoxPosX.valueChanged.connect(self.set_contour)
        self.ui.spinBoxPosY.valueChanged.connect(self.set_contour)
        self.ui.spinBoxRadius.valueChanged.connect(self.set_contour)
        self.ui.btnApplySnake.clicked.connect(self.apply_snake)
        self.ui.btnApplyHough.clicked.connect(self.apply_hough_transform)

    def on_tab_changed(self, index):
        """Handle tab widget changes"""
        current_tab = self.ui.tabWidget.currentWidget().objectName()
        
        if current_tab == "tabSnakeContour":
            self.init_snake_mode()
        elif current_tab == "tabHoughTransform":
            self.init_hough_mode()

    def init_snake_mode(self):
        if self.image is None:
            logging.warning("No image loaded for snake mode")
            return
            
        image = self.get_NumpyArray()
        pub.sendMessage("snake.initial", image=image)
        
        # Set initial contour position and radius based on image size
        if self.image is not None:
            width = self.image.width()
            height = self.image.height()
            # Set initial values to center of image
            self.ui.spinBoxPosX.setValue(width // 2)
            self.ui.spinBoxPosY.setValue(height // 2)
            # Set radius to 1/4 of the smallest dimension
            self.ui.spinBoxRadius.setValue(min(width, height) // 4)
            self.set_contour()

    def _init_modes(self):
        if "tabSnakeContour" == self.ui.tabWidget.currentWidget().objectName():
            self.init_snake_mode()
        elif "tabHough" == self.ui.tabWidget.currentWidget().objectName():
            self.init_hough_mode()
        elif "tabEdgeDetection" == self.ui.tabWidget.currentWidget().objectName():
            self.init_edge_detection_mode()

    def init_edge_detection_mode(self):
        """Initialize Edge Detection mode"""
        if self.image is None:
            logging.warning("No image loaded for Edge Detection mode")
            return

    def init_hough_mode(self):
        """Initialize Hough Transform mode"""
        if self.image is None:
            logging.warning("No image loaded for Hough Transform mode")
            return
            
        # Send the image to the Hough Transform processor
        image_np = self.get_NumpyArray()
        pub.sendMessage("hough.loadImage", image=image_np)
        
        # Set default UI values
        self.ui.comboShapeSelection.setCurrentText("Line")
        self.ui.sliderHoughThreshold.setValue(50)
        
        # Connect value display for threshold slider
        self.ui.sliderHoughThreshold.valueChanged.connect(
            lambda val: self.ui.labelThresholdValue.setText(str(val))
        )
        
        # Connect shape selection to parameter page switching
        self.ui.comboShapeSelection.currentTextChanged.connect(self.switch_hough_params_page)
        
        # Connect line parameter sliders to value labels
        self.ui.sliderMinLineLength.valueChanged.connect(
            lambda val: self.ui.labelMinLineLengthValue.setText(str(val))
        )
        self.ui.sliderMaxLineGap.valueChanged.connect(
            lambda val: self.ui.labelMaxLineGapValue.setText(str(val))
        )
        
        # Connect circle parameter sliders to value labels
        self.ui.sliderCircleMinRadius.valueChanged.connect(
            lambda val: self.ui.labelCircleMinRadiusValue.setText(str(val))
        )
        self.ui.sliderCircleMaxRadius.valueChanged.connect(
            lambda val: self.ui.labelCircleMaxRadiusValue.setText(str(val))
        )
        self.ui.sliderCircleMinDist.valueChanged.connect(
            lambda val: self.ui.labelCircleMinDistValue.setText(str(val))
        )
        
        # Connect ellipse parameter sliders to value labels
        self.ui.sliderEllipseMinArea.valueChanged.connect(
            lambda val: self.ui.labelEllipseMinAreaValue.setText(str(val))
        )
        self.ui.sliderEllipseMaxArea.valueChanged.connect(
            lambda val: self.ui.labelEllipseMaxAreaValue.setText(str(val))
        )
        
        # # Connect common parameter sliders to value labels
        # self.ui.sliderCannyLow.valueChanged.connect(
        #     lambda val: self.ui.labelCannyLowValue.setText(str(val))
        # )
        # self.ui.sliderCannyHigh.valueChanged.connect(
        #     lambda val: self.ui.labelCannyHighValue.setText(str(val))
        # )
        # self.ui.sliderBlurSize.valueChanged.connect(
        #     lambda val: self.ui.labelBlurSizeValue.setText(str(val))
        # )
        
        # Connect parameter sliders to update_hough_param
        # self.ui.sliderCannyLow.valueChanged.connect(self.update_hough_param)
        # self.ui.sliderCannyHigh.valueChanged.connect(self.update_hough_param)
        # self.ui.sliderBlurSize.valueChanged.connect(self.update_hough_param)
        self.ui.sliderMinLineLength.valueChanged.connect(self.update_hough_param)
        self.ui.sliderMaxLineGap.valueChanged.connect(self.update_hough_param)
        self.ui.sliderCircleMinRadius.valueChanged.connect(self.update_hough_param)
        self.ui.sliderCircleMaxRadius.valueChanged.connect(self.update_hough_param)
        self.ui.sliderCircleMinDist.valueChanged.connect(self.update_hough_param)
        self.ui.sliderEllipseMinArea.valueChanged.connect(self.update_hough_param)
        self.ui.sliderEllipseMaxArea.valueChanged.connect(self.update_hough_param)
        
        # Display correct parameter page for initial shape
        self.switch_hough_params_page(self.ui.comboShapeSelection.currentText())

    def update_hough_param(self):
        """Update Hough Transform parameters based on UI controls"""
        sender = self.sender()
        if sender is None:
            return
            
        # Map UI control names to parameter names
        param_map = {
            'sliderCannyLow': 'canny_low',
            'sliderCannyHigh': 'canny_high',
            'sliderBlurSize': 'blur_size',
            'sliderMinLineLength': 'line_min_length',
            'sliderMaxLineGap': 'line_max_gap',
            'sliderCircleMinRadius': 'circle_min_radius',
            'sliderCircleMaxRadius': 'circle_max_radius',
            'sliderCircleMinDist': 'circle_min_dist'
        }
        
        # Get the parameter name based on the sender object's name
        sender_name = sender.objectName()
        if sender_name in param_map:
            param_name = param_map[sender_name]
            value = sender.value()
            
            # Handle special cases
            if param_name == 'blur_size' and value % 2 == 0:
                # Blur size must be odd
                value += 1
                sender.setValue(value)
                
            # Send the parameter update
            pub.sendMessage("hough.updateParams", param_name=param_name, value=value)
            logging.debug(f"Updated {param_name} to {value}")

    def switch_hough_params_page(self, shape_type):
        """Switch the stacked widget page based on shape selection"""
        if shape_type == "Line":
            self.ui.stackedWidgetShapeParams.setCurrentIndex(0)
        elif shape_type == "Circle":
            self.ui.stackedWidgetShapeParams.setCurrentIndex(1)
        elif shape_type == "Ellipse":
            self.ui.stackedWidgetShapeParams.setCurrentIndex(2)

    def _setup_mouse_tracking(self):
        # Enable mouse tracking on the image label
        self.ui.ImageLabel.setMouseTracking(True)
        # Store original event handlers
        self.original_mouse_press = self.ui.ImageLabel.mousePressEvent
        self.original_mouse_move = self.ui.ImageLabel.mouseMoveEvent
        self.original_mouse_release = self.ui.ImageLabel.mouseReleaseEvent
        self.original_wheel_event = self.ui.ImageLabel.wheelEvent
        # Override with custom event handlers
        self.ui.ImageLabel.mousePressEvent = self._mouse_press_event
        self.ui.ImageLabel.mouseMoveEvent = self._mouse_move_event
        self.ui.ImageLabel.mouseReleaseEvent = self._mouse_release_event
        self.ui.ImageLabel.wheelEvent = self._wheel_event

    def _mouse_press_event(self, event):
        if self.image is None:
            return
            
        # Get mouse position in image coordinates
        pos = self._convert_pos_to_image_coords(event.position())
        if pos is None:
            return
            
        # Get current contour parameters
        center_x = self.ui.spinBoxPosX.value()
        center_y = self.ui.spinBoxPosY.value()
        radius = self.ui.spinBoxRadius.value()
        
        # Calculate distance from click to center
        distance = math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
        
        # If click is inside the circle or near its center, begin dragging
        if distance <= radius:
            self.dragging = True
            self.last_pos = pos
            logging.info(f"Starting center drag from ({center_x}, {center_y})")

    def _mouse_move_event(self, event):
        if self.image is None or not self.dragging:
            return
            
        pos = self._convert_pos_to_image_coords(event.position())
        if pos is None or self.last_pos is None:
            return
            
        # Update center position
        center_x = self.ui.spinBoxPosX.value()
        center_y = self.ui.spinBoxPosY.value()
        
        # Calculate the movement delta
        delta_x = pos[0] - self.last_pos[0]
        delta_y = pos[1] - self.last_pos[1]
        
        # Apply the movement to the center
        new_x = center_x + delta_x
        new_y = center_y + delta_y
        
        # Update the spinboxes (which will trigger set_contour)
        self.ui.spinBoxPosX.setValue(int(new_x))
        self.ui.spinBoxPosY.setValue(int(new_y))
        
        logging.debug(f"Dragging center to ({new_x}, {new_y})")
        
        # Update last position
        self.last_pos = pos

    def _mouse_release_event(self, event):
        if self.dragging:
            logging.info("Mouse drag ended")
            self.dragging = False
            self.last_pos = None

    def _wheel_event(self, event):
        """Handle mouse wheel events to adjust the contour radius"""
        if self.image is None:
            return
        
        # Get current radius
        current_radius = self.ui.spinBoxRadius.value()
        
        # Get wheel delta (typically +/- 120 per step)
        delta = event.angleDelta().y()
        
        # Determine increment amount (faster when holding Ctrl)
        increment = 5 if event.modifiers() & Qt.KeyboardModifier.ControlModifier else 1
        
        # Calculate new radius value
        if delta > 0:  # Scrolling up
            new_radius = current_radius + increment
        else:  # Scrolling down
            new_radius = current_radius - increment
        
        # Ensure minimum radius (not too small)
        new_radius = max(new_radius, 10)
        
        # Update the radius spinbox (which will trigger set_contour)
        self.ui.spinBoxRadius.setValue(new_radius)
        
        logging.debug(f"Adjusting radius using scroll wheel: {current_radius} -> {new_radius}")
        
        # Prevent event from being passed to parent widgets
        event.accept()

    def _convert_pos_to_image_coords(self, pos):
        """Convert screen coordinates to image coordinates"""
        if self.image is None:
            return None
            
        # Get label dimensions
        label_width = self.ui.ImageLabel.width()
        label_height = self.ui.ImageLabel.height()
        
        # Get image dimensions
        img_width = self.image.width()
        img_height = self.image.height()
        
        # If using setScaledContents(True), calculate the scaling ratios
        scale_x = img_width / label_width
        scale_y = img_height / label_height
        
        # Convert QPointF to x, y coordinates
        screen_x = pos.x()
        screen_y = pos.y()
        
        # Convert screen coordinates to image coordinates
        image_x = int(screen_x * scale_x)
        image_y = int(screen_y * scale_y)
        
        # Ensure coordinates are within image bounds
        image_x = max(0, min(image_x, img_width - 1))
        image_y = max(0, min(image_y, img_height - 1))
        
        return (image_x, image_y)
    
    def set_contour(self):
        center = (self.ui.spinBoxPosX.value(), self.ui.spinBoxPosY.value())
        radius = self.ui.spinBoxRadius.value()
        logging.info(f"Setting contour: center={center}, radius={radius}")
        pub.sendMessage("snake.setContour", center=center, radius=radius)
    
    def set_initial_contour(self, image_contour):
        self.image_output = self.from_ndarray_to_QPixmap(image_contour)
        self.display_image()

    def upload_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            self.image = QPixmap(filename)
            logging.info(f"Loaded image: {filename}, size: {self.image.width()}x{self.image.height()}")
            self.image_output = None
            self.display_image()
            self._init_modes()

    def get_QPixmap(self):
        return self.image
    
    def get_NumpyArray(self):
        if self.image is None:
            logging.error("Attempted to get NumPy array from None image")
            return None
            
        qpixmap_image = self.image
        qimage = qpixmap_image.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)
        return arr
    
    def get_QImage(self):
        if self.image is None:
            return None
        return self.image.toImage()
    
    def from_ndarray_to_QPixmap(self, arr):
        if arr is None:
            logging.error("Attempted to convert None array to QPixmap")
            return None
            
        height, width, channel = arr.shape
        bytesPerLine = 4 * width
        qimage = QImage(arr.data, width, height, bytesPerLine, QImage.Format.Format_RGBA8888)
        qpixmap = QPixmap.fromImage(qimage)
        return qpixmap
    
    def set_snake_output(self, image_result,perimeter,area,chain_code):
        if image_result is None:
            logging.error("Received None output for snake")
            return
        self.ui.btnApplySnake.setEnabled(True)
        self.clear_logging()
        pixmap = self.from_ndarray_to_QPixmap(image_result)
        self.image_output = pixmap
        self.display_image()
        logging.info("Updated snake output display")

    def set_hough_output(self, image_result):
        """Handle the Hough Transform processing result"""
        if image_result is None:
            logging.error("Received None output for Hough Transform")
            return
        
        # Re-enable the apply button
        self.ui.btnApplyHough.setEnabled(True)
        
        # Convert result to QPixmap and display
        pixmap = self.from_ndarray_to_QPixmap(image_result)
        self.image_output = pixmap
        self.display_image()
        logging.info("Updated Hough Transform output display")

    def display_image(self):
        if self.image_output is not None:
            self.ui.ImageLabel.setPixmap(self.image_output)
            self.ui.ImageLabel.setScaledContents(True)
            self.ui.ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.ImageLabel.show()
        elif self.image is not None:
            self.ui.ImageLabel.setPixmap(self.image)
            self.ui.ImageLabel.setScaledContents(True)
            self.ui.ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.ImageLabel.show()
        else:
            logging.error("No image to display")
            self.ui.ImageLabel.clear()
            self.ui.ImageLabel.setText("No Image")
            self.ui.ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
    def reset_image(self):
        self.image = None
        self.image_output = None
        self.ui.ImageLabel.clear()
        self.ui.ImageLabel.setText("No Image")
        self.ui.ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logging.info("Image reset")

    def apply_snake(self):
        """Apply snake contour algorithm with current parameters"""
        try:
            # Get parameter values from UI controls
            alpha = self.ui.sliderAlpha.value() / 100
            beta = self.ui.sliderBeta.value() / 100
            gamma = self.ui.sliderGamma.value() / 100
            iterations = self.ui.spinBoxIterations.value()
            pos_x = self.ui.spinBoxPosX.value()
            pos_y = self.ui.spinBoxPosY.value()
            radius = self.ui.spinBoxRadius.value()
            
            logging.info(f"Applying snake: alpha={alpha}, beta={beta}, gamma={gamma}, "
                         f"iterations={iterations}, pos=({pos_x},{pos_y}), radius={radius}")
            
            # Send message to snake processor to apply the algorithm
            self.ui.btnApplySnake.setEnabled(False)
            pub.sendMessage("snake.apply", 
                            alpha=alpha, 
                            beta=beta, 
                            gamma=gamma, 
                            iterations=iterations, 
                            center=(pos_x, pos_y), 
                            radius=radius)
        except Exception as e:
            logging.error(f"Error applying snake algorithm: {e}")

    def add_logging(self, message):
        self.ui.loggingLabel.setText(message)
    
    def clear_logging(self):
        self.ui.loggingLabel.setText("")

    def apply_hough_transform(self):
        """Apply Hough Transform based on user-selected parameters."""
        if self.image is None:
            logging.warning("No image loaded for Hough Transform")
            pub.sendMessage("ui.logging", message="Error: No image loaded")
            return

        # Get basic parameters
        shape_type = self.ui.comboShapeSelection.currentText()
        threshold = self.ui.sliderHoughThreshold.value()
        
        logging.info(f"Requesting Hough Transform: shape={shape_type}, threshold={threshold}")
        
        # Disable button while processing
        self.ui.btnApplyHough.setEnabled(False)
        pub.sendMessage("ui.logging", message="Processing...")
        
        try:
            # First update all the parameters individually
            # Common parameters for all shapes
            if hasattr(self.ui, 'sliderCannyLow'):
                pub.sendMessage("hough.updateParams", 
                               param_name="canny_low", 
                               value=self.ui.sliderCannyLow.value())
            
            if hasattr(self.ui, 'sliderCannyHigh'):
                pub.sendMessage("hough.updateParams", 
                               param_name="canny_high", 
                               value=self.ui.sliderCannyHigh.value())
            
            if hasattr(self.ui, 'sliderBlurSize'):
                blur_size = self.ui.sliderBlurSize.value()
                if blur_size % 2 == 0:
                    blur_size += 1
                pub.sendMessage("hough.updateParams", 
                               param_name="blur_size", 
                               value=blur_size)
            
            # Shape-specific parameters
            if shape_type == "Line":
                if hasattr(self.ui, 'sliderMinLineLength'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="line_min_length", 
                                   value=self.ui.sliderMinLineLength.value())
                
                if hasattr(self.ui, 'sliderMaxLineGap'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="line_max_gap", 
                                   value=self.ui.sliderMaxLineGap.value())
            
            elif shape_type == "Circle":
                if hasattr(self.ui, 'sliderCircleMinRadius'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="circle_min_radius", 
                                   value=self.ui.sliderCircleMinRadius.value())
                
                if hasattr(self.ui, 'sliderCircleMaxRadius'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="circle_max_radius", 
                                   value=self.ui.sliderCircleMaxRadius.value())
                
                if hasattr(self.ui, 'sliderCircleMinDist'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="circle_min_dist", 
                                   value=self.ui.sliderCircleMinDist.value())
            
            elif shape_type == "Ellipse":
                if hasattr(self.ui, 'sliderEllipseMinArea'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="ellipse_min_area", 
                                   value=self.ui.sliderEllipseMinArea.value())
                
                if hasattr(self.ui, 'sliderEllipseMaxArea'):
                    pub.sendMessage("hough.updateParams", 
                                   param_name="ellipse_max_area", 
                                   value=self.ui.sliderEllipseMaxArea.value())
            
            # Now send the process command with just the basic parameters
            pub.sendMessage("hough.apply", shape_type=shape_type, threshold=threshold)
            
        except Exception as e:
            logging.error(f"Error in apply_hough_transform: {e}")
            self.ui.btnApplyHough.setEnabled(True)
            pub.sendMessage("ui.logging", message=f"Error: {str(e)}")
