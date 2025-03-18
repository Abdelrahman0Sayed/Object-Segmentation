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
        pub.subscribe(self.add_logging, "ui.logging")
        pub.subscribe(self.clear_logging, "ui.clearLogging")

    def _bind_ui_events(self):
        self.ui.btnLoadImage.clicked.connect(self.upload_image)
        self.ui.btnReset.clicked.connect(self.reset_image)
        self.ui.tabWidget.currentChanged.connect(self.init_snake_mode)
        self.ui.spinBoxPosX.valueChanged.connect(self.set_contour)
        self.ui.spinBoxPosY.valueChanged.connect(self.set_contour)
        self.ui.spinBoxRadius.valueChanged.connect(self.set_contour)
        self.ui.btnApplySnake.clicked.connect(self.apply_snake)
        
    def _init_modes(self):
        if "tabSnakeContour" == self.ui.tabWidget.currentWidget().objectName():
            self.init_snake_mode()

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