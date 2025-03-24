import sys
from PyQt6.QtWidgets import QApplication
from MainWindowUI import MainWindowUI
import asyncio
from qasync import QEventLoop
from snake_2 import SnakeProcessor
from Hough_transform import HoughTransform
from Canny import CannyFilter

async def main():
    snake = SnakeProcessor()
    hough = HoughTransform()
    canny = CannyFilter()
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindowUI()
    window.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)