import cv2
import numpy as np 
import glob
import os.path as path 
import pickle
from config import parameters

def cam_cal(calibration_images_dir):
    '''
    calibrates the camera based on a given calibration chessboard images

    returns:
    - ret: bool stating whether there're returned values or not
    - mtx: camera matrix that transforms 3D objects to 2D image points
    - dst: distortion coeffitionts
    - rvecs: rotation vectors
    - tvecs: translation vectors
    '''

    '''
    object points are grid fillers that represent the number of inside corners in x & 
     y dimensions of the input chessboard image. These points are collected so that,
     for each input chessboard image, the actual points location in each image (img_pts)
     are compared, in a certain magical way, to this grid to clibrate the camera in the 
     cv2.calibrateCamera(obj_pts, img_pts, ...) function.
    the (6, 9), as said, are the number of inside corners in both dimensions. These numbers
     are put by hand depending on the used chessboard image
    '''
    obj_p= np.zeros((6*9, 3), np.float32)
    obj_p[:, :2]= np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    #store object & images points here:
    obj_pts= []
    img_pts= []

    '''
    read the chessboard images and extract the inside corner points
    '''
    image_files= glob.glob(path.join(calibration_images_dir, "calibration*.jpg"))
    for image_file in image_files:

        image= cv2.imread(image_file)
        gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #try to find chessboard inside corner points (img_pts) for this actual image
        #if successful, save it and a corresponding obj_pts to use them for calibration
        found, img_p= cv2.findChessboardCorners(gray, (9, 6), None)
        if found:
            img_pts.append(img_p)
            obj_pts.append(obj_p)
        
    return cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

def undistort(image):

    dist_coeffs= parameters["dist_coeffs"]
    cam_mtx= parameters["cam_mtx"]

    return cv2.undistort(image, cam_mtx, dist_coeffs, newCameraMatrix= cam_mtx)

def thresh_image_in_HSV(image):
    yellow_thresh_min= parameters["yellow_thresh_min"]
    yellow_thresh_max= parameters["yellow_thresh_max"]    

    HSV= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #applying (yellow_thresh_min< HSV) && (yellow_thresh_max> HSV)
    first_cond= np.all(HSV> yellow_thresh_min, axis=2)
    second_cond= np.all(HSV< yellow_thresh_max, axis=2)
    return np.logical_and(first_cond, second_cond)

def magic(image):
    '''
    apply histogram equalizer and threshold the result 
    '''
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized= cv2.equalizeHist(gray)
    _, th= cv2.threshold(equalized, thresh= 250, maxval= 255, type= cv2.THRESH_BINARY)
    return th

def calculate_sobel(image):
    sobel_kernel= parameters["sobel_kernel"]

    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    #_, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)
    #return sobel_mag.astype(bool)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    return sobel_mag

def binarize(image):
    '''
    in this function we transform the 3 channel image to a 1 channel image
    during this transformation, we'll keep important details that might contain 
    the lanes inside them. We keep white and yellow colors (and change them to white)
    ,we also keep the strong gradiants, and finally, we clean the image to fill gaps (holes)
    inside the image
    ''' 

    #the first 2 dimensions in image (height, width, )
    image_h, image_w= image.shape[:2]

    #one channel placeholder (because binarized image will have only 1 channel)
    binary_image= np.zeros(shape= (image_h, image_w), dtype= np.uint8)

    #change the color space of the image on the fly to HSV to extract the yellow color
    #and return its location in a mask to be (ORed) to the binary image
    yellow_mask= thresh_image_in_HSV(image)
    binary_image= np.logical_or(binary_image, yellow_mask)

    #extract white color using magic
    white_mask= magic(image)
    binary_image= np.logical_or(binary_image, white_mask)

    #calculate sobel and OR it to the binary_image
    sobel_mask= calculate_sobel(image)
    binary_image= np.logical_or(binary_image, sobel_mask)

    #apply morphy_close to fill holes inside the binary_image
    morphy_kernel= np.ones((5,5), np.uint8)
    binary_image= cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, morphy_kernel)

    return binary_image 

def bird_eye(image):
    '''
    apply perpective transformation
    '''
    #the first 2 dimensions in image (height, width, )
    image_h, image_w= image.shape[:2]

    #define the two boxes, the src and the dst box
    src = np.float32([[image_w, image_h-10],    # br
                      [0, image_h-10],    # bl
                      [546, 460],   # tl
                      [732, 460]])  # tr
    dst = np.float32([[image_w, image_h],       # br
                      [0, image_h],       # bl
                      [0, 0],       # tl
                      [image_w, 0]])      # tr
    
    #get the matrix that transforms from src box to dst box, and its inverse
    matrix= cv2.getPerspectiveTransform(src, dst)
    matrix_inv= cv2.getPerspectiveTransform(dst, src)

    #apply the transformation matrix from src to dst
    transformed= cv2.warpPerspective(image , matrix, (image_w, image_h), flags=cv2.INTER_LINEAR)

    return transformed, matrix, matrix_inv 