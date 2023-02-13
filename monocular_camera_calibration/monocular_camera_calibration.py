# version: 1.0
# name: monocular_camera_calibration.py
# python version: 3.10.6
# author: mikehaku
# date: 2023-02-13

# this script is used to calibrate the monocular camera
# the calibration images are stored in the folder 'calibration_img'
# the undistorted images are stored in the folder 'undistorted_img'
# the intrinsic matrix and distortion coefficients are stored in the npz file 'calibration.npz'

import cv2 as cv
import numpy as np
import glob as gb

# this function is used to find the corners of the chessboard
def find_corners(img, nx, ny):
    # convert the image to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # find the corners of the chessboard
    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)
    # refine the corners
    if ret == True:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), \
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Optional:
        # draw the corners
        # cv.drawChessboardCorners(img, (nx, ny), corners, ret)
        # # show the result
        # cv.imshow('corners', img)
        # cv.waitKey(0)
    return ret, corners

# this function is used to calibrate the camera
# nx and ny are the number of corners in the x and y direction
def calibrate_camera(nx, ny):
    # prepare the object points
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # add scale factor for the object points
    objp = objp * 20
    # prepare the object points and image points
    objpoints = []
    imgpoints = []
    # read the images
    images = gb.glob('calibration_img' + '/*.png')
    # find the corners of the chessboard
    for fname in images:
        img = cv.imread(fname)
        ret, corners = find_corners(img, nx, ny)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

# this function is used to undistort the image
def undistort_image(img, mtx, dist):
    undist = cv.undistort(img, mtx, dist, None, mtx)
    return undist

# main function
if __name__ == '__main__':
    
    # corners of the chessboard
    nx = 11
    ny = 8

    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(nx, ny)

    # show the intrinsic matrix and distortion coefficients
    print('mtx = ', mtx)
    print('dist = ', dist)

    # show reprojection error
    print('reprojection error = ', ret)

    # hold the terminal
    input('Press Enter to continue...')

    # the folder where the calibration images are stored
    calib_img_folder = 'calibration_img/'

    # the folder where the undistorted images are stored
    undist_img_folder = 'undistorted_img/'
    
    # read the images
    images = gb.glob(calib_img_folder + '*.png')
    
    # undistort the image
    for fname in images:

        # for each image, undistort it
        img = cv.imread(fname)
        undist = undistort_image(img, mtx, dist)

        # #show the result
        # cv.imshow('undist', undist)
        # cv.waitKey(0)

        # take fname from the path
        fname = fname.split('/')[-1]        

        # save the result
        cv.imwrite(undist_img_folder + 'undist_' + fname, undist)

    # save the intrinsic matrix and distortion coefficients 
    # and the reprojection error in a npz file
    np.savez('calibration.npz', mtx=mtx, dist=dist, ret=ret)

    # save the intrinsic matrix and distortion coefficients 
    # in a file which can be read by cpp
    with open('calibration.txt', 'w') as f:
        f.write('mtx = ')
        for i in range(mtx.shape[0]):
            for j in range(mtx.shape[1]):
                f.write(str(mtx[i, j]))
                if j != mtx.shape[1] - 1:
                    f.write(', ')
        f.write('.\n')
        f.write('dist = ')
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                f.write(str(dist[i, j]))
                if j != dist.shape[1] - 1:
                    f.write(', ')
        f.write('.\n')
        f.write('reprojection error = ')
        f.write(str(ret))
        f.write('.\n')

    # save the intrinsic matrix and distortion coefficients in a yaml file
    import yaml
    data = dict(
        mtx = mtx.tolist(),
        dist = dist.tolist(),
        ret = ret
    )
    with open('calibration.yaml', 'w') as f:
        yaml.dump(data, f)

    # information for saving
    print('Calibration data saved.')

    # Optional:
    # read intrinsic matrix and distortion coefficients from the npz file
    # with np.load('calibration.npz') as X:
    #     mtx, dist = [X[i] for i in ('mtx', 'dist')]
    #     # show the intrinsic matrix and distortion coefficients 
    #     # and the reprojection error in the terminal and hold the terminal
    #     print('mtx = ', mtx)
    #     print('dist = ', dist)
    #     print('reprojection error = ', ret)
    #     input('Press Enter to continue...')




