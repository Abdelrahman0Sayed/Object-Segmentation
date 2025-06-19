# Object-Segmentation
## Overview

The PyQt6 application allows users to load images, apply filters, and adjust parameters for object segmentation using Active Contour (snakes), Canny Edge Detection, and Hough Transform for line, circle, and ellipse detection.

## UI

Interactive interface with tabs for Active Contour, Hough Transform, and Edge Detection, supporting real-time adjustments and visualizations.

## Snake Contour

Dynamic contour evolution to detect object boundaries.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| Snake-Active-Contour | ![Snake-Active-Contour](https://github.com/user-attachments/assets/e5fab0cb-01ca-4316-8964-045104977918) | Evolves an initial contour using internal (elasticity, stiffness), external (gradient), and balloon energies, with Cython optimization for 1000 iterations. |

## Edge Detection

Highlights significant edges in images.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| Canny-Edge-Detection | ![Canny-Edge-Detection](https://github.com/user-attachments/assets/3f798a81-7aff-4702-9cd1-a05f42698f1d) | Uses Gaussian smoothing, Sobel gradients, non-maximum suppression, and double thresholding with hysteresis to detect edges. |

## Hough Transform

Detects lines, circles, and ellipses in edge images.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| Hough-Transform-Line | ![Hough-Transform-Line](https://github.com/user-attachments/assets/91dd200e-d211-4c1e-89d7-42accd337955) | Implements Hough Transform to detect lines using an accumulator and peak detection, optimized with Cython. |
| Hough-Transform-Circle | ![Hough-Transform-Circle](https://github.com/user-attachments/assets/62152a8b-1c63-43ad-886e-d25709eef369) | Detects circles using a 3D accumulator and peak detection, with adjustable radius constraints. |
| Hough-Transform-Ellipse | ![Hough-Transform-Ellipse](https://github.com/user-attachments/assets/f70348b3-4f3f-422f-9418-99b83513dd36) | Fits ellipses to contours with area and point filters, using OpenCV’s ellipse fitting function. |

## Getting Started

1. Clone the repository: `git clone https://github.com/Abdelrahman0Sayed/Object-Segmentation`
2. Install dependencies (e.g., PyQt6, NumPy, OpenCV, Cython).
3. Run the UI to explore the techniques.

## Contributors

<table>
  <tr>
            <td align="center">
      <a href="https://github.com/Abdelrahman0Sayed">
        <img src="https://avatars.githubusercontent.com/u/113141265?v=4" width="250px;" alt="Abdelrahman Sayed"/>
        <br />
        <sub><b>Abdelrahman Sayed</b></sub>
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/salahmohamed03">
        <img src="https://avatars.githubusercontent.com/u/93553073?v=4" width="250px;" alt="Salah Mohamed"/>
        <br />
        <sub><b>Salah Mohamed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ayatullah-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125223938?v=" width="250px;" alt="Ayatullah Ahmed"/>
        <br />
        <sub><b>Ayatullah Ahmed</b></sub>
      </a>
    </td>
        </td>
        <td align="center">
      <a href="https://github.com/AhmeedRaafatt">
        <img src="https://avatars.githubusercontent.com/u/125607744?v=4" width="250px;" alt="Ahmed Raffat"/>
        <br />
        <sub><b>Ahmed Rafaat</b></sub>
      </a>
    </td>
  </tr>
</table>
