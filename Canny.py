import cv2 
import numpy as np
from pubsub import pub
import concurrent.futures
import asyncio
        


class CannyFilter :
    def __init__(self):
        self.img_original = None
        self.img_copy = None
        self._bind_events()
    
        
    def _bind_events(self):
        pub.subscribe(self.cannyOperator, "detect edges")

    """
    Canny Edge Detector
    Steps:
    1- Apply Gaussian Filter to remove noise effect
    2- Apply Gradient Operator(Sobel Filter) and get magnitude and orientation
    3- non-maximum supperession : transform edges into thin edges by keep the local maxima only
    4- Double Thresholding & Hystersis: Keep strong edges and weaks edges connected to it and remove the rest.
    """ 
    def cannyOperator(self, image, sigmaValue, highThreshold, lowThreshold):
        # Define Paramters
        kernelSize= (5, 5)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First : Apply Gaussian Filter.
        filteredImage = cv2.GaussianBlur(grayImage, kernelSize, sigmaX=sigmaValue) 
        # Second : Apply Gradient Operator + Double Thresholding.
        edgyImage, orientation, x_edges_image, y_edges_image = self.gradientOperator(filteredImage)
        # Third : Apply non-maximum supperssion.
        edgyImage = self.nonMaximumSuppression(edgyImage, orientation)
        # Fourth : Apply Double Thresholding and Hystersis.
        edgyImage = self.DoubleThresholdingAndHysteresis(edgyImage, highThreshold, lowThreshold)


        # Publish the result to the main window
        pub.sendMessage("canny.output", image=edgyImage)

    def cannyOperatorNOGUI(self, image, sigmaValue, highThreshold, lowThreshold):
        # Define Paramters
        kernelSize= (5, 5)
        # Check if the image is already grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayImage = image

        # First : Apply Gaussian Filter.
        filteredImage = cv2.GaussianBlur(grayImage, kernelSize, sigmaX=sigmaValue) 
        # Second : Apply Gradient Operator + Double Thresholding.
        edgyImage, orientation, x_edges_image, y_edges_image = self.gradientOperator(filteredImage)
        # Third : Apply non-maximum supperssion.
        edgyImage = self.nonMaximumSuppression(edgyImage, orientation)
        # Fourth : Apply Double Thresholding and Hystersis.
        edgyImage = self.DoubleThresholdingAndHysteresis(edgyImage, highThreshold, lowThreshold)

        return edgyImage
        

    def gradientOperator(self, filteredImage):
        # Gradient Operator using a soble kernel as the first derivative mask.
        sobelKernel_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobelKernel_X = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        kernels = {
            "KernelX" : sobelKernel_X,
            "KernelY" : sobelKernel_Y
        }
        # return applyKernelForLoop(filteredImage, kernels)
        return self.applyKernelFT(filteredImage, kernels)

    """
    Apply Kernels to the images and return the Horizontal edges, Vertical edges and filtered image
    """ 
    def applyKernelForLoop(self, image, kernels):
        pass

    def applyKernelFT(self, image, kernels):
        height, width = image.shape  

        kernelX_padded = np.zeros((height, width), dtype=np.float32)
        kernelY_padded = np.zeros((height, width), dtype=np.float32)

        kernelX_padded[:3, :3] = kernels['KernelX']
        kernelY_padded[:3, :3] = kernels['KernelY']

        kernelX = np.fft.fft2(kernelX_padded)
        kernelY = np.fft.fft2(kernelY_padded)
        image_FT = np.fft.fft2(image)

        Gx_edges = image_FT * kernelX
        Gy_edges = image_FT * kernelY

        x_edges_image = np.fft.ifft2(Gx_edges).real
        y_edges_image = np.fft.ifft2(Gy_edges).real
        #x_edges_image = cv2.normalize(np.abs(x_edges_image), None, 0, 255, cv2.NORM_MINMAX)
        #y_edges_image = cv2.normalize(np.abs(y_edges_image), None, 0, 255, cv2.NORM_MINMAX)

        edgyImage = np.sqrt(x_edges_image ** 2 + y_edges_image ** 2)
        orientation = np.angle(image_FT) 
    

        #edgyImage = cv2.normalize(np.abs(edgyImage), None, 0, 255, cv2.NORM_MINMAX)

        return edgyImage, orientation, x_edges_image, y_edges_image


    """
    This function used to give a thinner edges by check the pixel and its neighbors 
    (and its neighobrs depends on the direction of the edge) and if it's the local maxima 
    i will keep it, otherwise it will be set = 0
    """
    def nonMaximumSuppression(self, magnitude, orientation):
        h, w = magnitude.shape
        suppressed = np.zeros((h, w), dtype=np.float32)
        
        # Convert angles from radians to degrees
        angle = np.rad2deg(orientation) % 180

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                q, r = 255, 255  # Neighbor pixels
                # Choose the neighbors depending on the direction of edge in this pixel.            
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1] 
                    r = magnitude[i, j - 1]  
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i - 1, j + 1] 
                    r = magnitude[i + 1, j - 1]  
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i - 1, j]  
                    r = magnitude[i + 1, j] 
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]  
                    r = magnitude[i + 1, j + 1]  

                # Keep the pixel if it's the local maximum
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed


    """
    Function used for double thresholding (Low Threshold, High Threshold) Keep strong Pixel
    and Weak Pixels that are connected to Strong pixels, otherwise will be removed.
    """
    def DoubleThresholdingAndHysteresis(self, img, lowThreshold, highThreshold):
        h, w = img.shape
        strongPixel = 255
        weakPixel = 70  

        strong_edges = (img >= highThreshold)
        weak_edges = (img < highThreshold) & (img >= lowThreshold)

        output = np.zeros_like(img, dtype=np.uint8)
        output[strong_edges] = strongPixel
        output[weak_edges] = weakPixel 

        # Connect weak edges to strong edges
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if output[i, j] == weakPixel:
                    if np.any(output[i-1:i+2, j-1:j+2] == strongPixel):
                        output[i, j] = strongPixel  
                    else:
                        output[i, j] = 0

        return output

