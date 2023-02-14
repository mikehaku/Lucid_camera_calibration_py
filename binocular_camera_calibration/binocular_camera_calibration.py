'''
    This script is used to calibrate the binocular camera.
    Since I do not have a binocular camera, I used the images from the internet.
    The images are obtained from the following link:
    https://github.com/niconielsen32/ComputerVision/tree/master/StereoVisionDepthEstimation/images
    Thanks to the author of the link, and the images are used for educational purpose only.
    If you want to use the images for other purposes, please contact the author of the link.
'''

import numpy as np
import cv2 as cv
import glob as gb

################# 1. prepare the images and chessboard information #################

# the directory to the images
img_folder = 'nikolai_binocular_imgs/'

# the directory to the images from the left camera
left_img_folder = img_folder + 'left/'

# the directory to the images from the right camera
right_img_folder = img_folder + 'right/'

# read the images from two cameras' folders
left_images = gb.glob( left_img_folder + '*.png')
right_images = gb.glob( right_img_folder + '*.png')

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 30 is the maximum number of iterations
# 0.001 is the minimum accuracy

# input the size of chessboard
width = 9
height = 6

# input the scale of the chessboard
scale = 20

# save the image size for later usage in the calibration
gray_shape = cv.cvtColor(cv.imread(left_images[0]),cv.COLOR_BGR2GRAY).shape[::-1]

# find the chessboard corners in the images
def find_corners(images):
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * scale

    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv.drawChessboardCorners(img, (width, height), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    return objpoints, imgpoints

################# 2. calibrate the two cameras #################

# find the corners in the images from the left camera
objpoints_left, imgpoints_left = find_corners(left_images)

# find the corners in the images from the right camera
objpoints_right, imgpoints_right = find_corners(right_images)

# calibrate the left camera
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv.calibrateCamera(objpoints_left, imgpoints_left, gray_shape, None, None)

# calibrate the right camera
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv.calibrateCamera(objpoints_right, imgpoints_right, gray_shape, None, None)

################# 3. calibrate the stereo camera #################

# stereo calibration
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
flags |= cv.CALIB_FIX_PRINCIPAL_POINT
flags |= cv.CALIB_USE_INTRINSIC_GUESS
flags |= cv.CALIB_FIX_FOCAL_LENGTH
flags |= cv.CALIB_FIX_ASPECT_RATIO
flags |= cv.CALIB_ZERO_TANGENT_DIST
flags |= cv.CALIB_RATIONAL_MODEL
flags |= cv.CALIB_SAME_FOCAL_LENGTH
flags |= cv.CALIB_FIX_K3
flags |= cv.CALIB_FIX_K4
flags |= cv.CALIB_FIX_K5

# termination criteria
criteria_stereo = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
# 100 is the maximum number of iterations
# 1e-5 is the minimum accuracy

# stereo calibration
ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F\
     = cv.stereoCalibrate(objpoints_left, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, \
        gray_shape, criteria=criteria_stereo, flags=flags)

# save the results
np.savez('binocular_camera.npz', ret=ret, mtx_left=mtx_left, dist_left=dist_left, mtx_right=mtx_right, dist_right=dist_right, R=R, T=T, E=E, F=F)

################# 4. rectify the images #################

# rectify the images
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, gray_shape, R, T)

# save the results
np.savez('binocular_camera_rectify.npz', R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2)

################# 5. undistort the images #################

# undistort the images from the left camera
for fname in left_images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    # newcameratx must be obtained to undistort the image
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx_left, dist_left, (w, h), 1, (w, h))
    dst = cv.undistort(img, mtx_left, dist_left, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('left_rectify/' + fname[(len(img_folder) + 5):], dst)

# undistort the images from the right camera
for fname in right_images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx_right, dist_right, (w, h), 1, (w, h))
    dst = cv.undistort(img, mtx_right, dist_right, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('right_rectify/' + fname[(len(img_folder) + 6):], dst)

################# 6. save the calibration results

# save all calibration results in a readable file which can also be used by cpp
with open('binocular_camera.txt', 'w') as f:
    f.write('ret: ' + str(ret) + '\n')
    f.write('mtx_left: ' + str(mtx_left) + '\n')
    f.write('dist_left: ' + str(dist_left) + '\n')
    f.write('mtx_right: ' + str(mtx_right) + '\n')
    f.write('dist_right: ' + str(dist_right) + '\n')
    f.write('R: ' + str(R) + '\n')
    f.write('T: ' + str(T) + '\n')
    f.write('E: ' + str(E) + '\n')
    f.write('F: ' + str(F) + '\n')
    f.write('R1: ' + str(R1) + '\n')
    f.write('R2: ' + str(R2) + '\n')
    f.write('P1: ' + str(P1) + '\n')
    f.write('P2: ' + str(P2) + '\n')
    f.write('Q: ' + str(Q) + '\n')
    f.write('roi1: ' + str(roi1) + '\n')
    f.write('roi2: ' + str(roi2) + '\n')
# close the file
f.close()