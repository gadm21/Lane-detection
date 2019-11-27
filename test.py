import cv2
import glob

from Line import Line
from camera_generals import cam_cal, bird_eye, undistort, binarize
from general_helper import get_fit_by_sliding_windows
from config import parameters


if __name__ == '__main__':

    line_lt, line_rt = Line(buffer_len=10), Line(buffer_len=10)

    ret, mtx, dist, rvecs, tvecs = cam_cal(calibration_images_dir='camera_cal')
    parameters["cam_mtx"]= mtx
    parameters["dist_coeffs"]= dist
    
    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img)

        img_binary = binarize(img_undistorted)

        img_birdeye, M, Minv = bird_eye(img_binary)

        line_lt, line_rt, img_out = get_fit_by_sliding_windows(img_birdeye, line_lt, line_rt, parameters)
        
        cv2.imshow("r", img_out)
        cv2.waitKey(0)
        cv2.destroyWindow("r")
