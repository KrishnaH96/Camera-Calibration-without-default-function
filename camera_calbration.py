import numpy as np
import cv2 as cv
from scipy.linalg import rq

image_x = np.array([757,758,758,759,1190,329,1204,340])
image_y = np.array([213,415,686,966,172,1041,850,159])

real_x = np.array([0,0,0,0,7,0,7,0])
real_y = np.array([0,3,7,11,1,11,9,1])
real_z = np.array([0,0,0,0,0,7,0,7])

A = []
for i in range(0,len(image_y)):

    A.append([0,0,0,0, -real_x[i], -real_y[i], -real_z[i],-1, image_y[i]*real_x[i], image_y[i]*real_y[i], image_y[i]*real_z[i], image_y[i]])
    A.append([real_x[i], real_y[i], real_z[i], 1,0,0,0,0, -image_x[i]*real_x[i], -image_x[i]*real_y[i], -image_x[i]*real_z[i],-image_x[i]])

A = np.array(A)

U, S, Vt = np.linalg.svd(A)

projection_matrix_camera = Vt[-1, : ].reshape((3,4))

M = projection_matrix_camera[:, :3]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")
print("The Projection Matrix for the camera calibration:   \n")
print(projection_matrix_camera)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")

camera_intrinsic_matrix_unnormalised, Rotation_matrix_camera = rq(M)


camera_intrinsic_matrix = camera_intrinsic_matrix_unnormalised/camera_intrinsic_matrix_unnormalised[(2,2)]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")
print("The intrinsic matrix of the camera:   \n")
print(camera_intrinsic_matrix)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")
print("The Rotation matrix between the world and camera frame :   \n")
print(Rotation_matrix_camera)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")

camera_intrinsic_matrix_inverse = np.linalg.pinv(camera_intrinsic_matrix)

camera_extrinsic_matrix = camera_intrinsic_matrix_inverse @ projection_matrix_camera

camera_translation_matrix = camera_extrinsic_matrix[:, 3]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")
print("The Translation matrix between the world and camera frame :   \n")
print(camera_translation_matrix)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")

reprojection_error_array= []

for i in range(len(real_x)):

    real_world_vector = np.array([real_x[i], real_y[i], real_z[i], 1])

    homogenous_image_coordinate= projection_matrix_camera @ real_world_vector

    homogenous_image_coordinate= homogenous_image_coordinate/homogenous_image_coordinate[2]

    new_image_x = homogenous_image_coordinate[0]
    new_image_y = homogenous_image_coordinate[1]

    given_image_point = np.array([image_x[i],image_y[i]])
    new_image_point = np.array([new_image_x,new_image_y])
  
    error = np.linalg.norm(given_image_point-new_image_point, ord=2)

    print(f" The reprojection error for {i+1}th point is {error}.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")

    reprojection_error_array.append(error)

mean_reprojection_error = np.mean(reprojection_error_array)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")
print(f"Mean error: {mean_reprojection_error} \n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: \n")


















