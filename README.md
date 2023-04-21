# Camera-Calibration-without-default-function


# Camera Calibration Pipeline

This pipeline outlines the steps required to calibrate a camera using a sample image of calibration points. The following libraries will be used: NumPy, OpenCV, and SciPy's Linear Algebra module.

## Step 1: Import Libraries

```
import numpy as np
import cv2
from scipy.linalg import svd
```

## Step 2: Define Calibration Points

Define the image and real-world coordinates of the calibration points as NumPy arrays. A minimum of six points are needed for camera calibration, but in this problem, we are given eight points.

<img width="725" alt="image" src="https://user-images.githubusercontent.com/113392023/233718915-482125aa-3082-4f9a-946e-f12f28489bcb.png">

## Step 3: Compute Projection Matrix

Compute the projection matrix P using below AP = 0, the mathematics for which will be explained in detail in the next question.

## Step 4: Populate Point List

Create an empty list called 'A' and use a for loop to populate it with the corresponding calibration point coordinates.

```
A = []
for i in range(num_points):
    u, v = img_pts[i]
    x, y, z = obj_pts[i]
    A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
    A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
```

## Step 5: Compute Projection Matrix using SVD

Use NumPy's Singular Value Decomposition (SVD) function to compute the projection matrix of the camera from the matrix A.

```
u, s, vh = np.linalg.svd(A)
P = vh[-1].reshape(3, 4)
```

## Step 6: Extract Camera Intrinsic Matrix and Rotation Matrix

Extract the camera intrinsic matrix and the rotation matrix of the camera from the projection matrix using the RQ decomposition.

```
K, R = rq(P[:, :3])
```

## Step 7: Normalize Intrinsic Matrix

Normalize the intrinsic matrix and print it along with the projection and rotation matrices.

```
K /= K[2, 2]
print("Intrinsic Matrix:\n", K)
print("Projection Matrix:\n", P)
print("Rotation Matrix:\n", R)
```

## Step 8: Compute Camera Extrinsic Matrix and Translation Matrix

Compute the camera extrinsic matrix and the translation matrix from the intrinsic matrix and projection matrix.

```
T = np.linalg.inv(-K @ P[:, :3]) @ P[:, 3]
```

## Step 9: Compute Reprojection Error

Define an empty list called 'reprojection_error_array' and use a for loop to compute the reprojection error for each calibration point using the projection matrix and the real-world coordinates of the calibration points.

```
reprojection_error_array = []
for i in range(num_points):
    X = np.array(obj_pts[i] + [1])
    x_est = P @ X
    x_est /= x_est[2]
    error = np.linalg.norm(x_est[:2] - img_pts[i])
    reprojection_error_array.append(error)
```

## Step 10: Compute Mean Reprojection Error

Compute the mean reprojection error from the list of reprojection errors.

```
mean_reprojection_error = np.mean(reprojection_error_array)
```

## Step 11: Print Mean Reprojection Error

Print the mean reprojection error.

```
print("Mean Reprojection Error: ", mean_reprojection_error)
```

By following these steps, you can calibrate your camera and obtain the intrinsic and extrinsic parameters needed for image

## Installation

To run the code in this repository, you need to have the following packages installed:

- OpenCV
- NumPy
- scipy

You can install the required packages using pip:

```
pip install opencv-python numpy
```
## Usage

To use this code, follow these steps:

1. Clone this repository: `[https://github.com/KrishnaH96/Visual-Odometry.git](https://github.com/KrishnaH96/Camera-Calibration-without-default-function.git)`
2. Run the Python script.
3. The program will display Projection matrix, Intrinsic Matrix of the camera, Rotation and Translation matrix between the camera frame and the world frame along with mean reprojection error.

## Contributing

If you find any issues with the code or want to suggest improvements, feel free to open an issue or create a pull request in this repository. 

