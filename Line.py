import numpy as np
import cv2
import glob
import collections
import matplotlib.pyplot as plt 
from camera_generals import cam_cal, undistort, binarize, bird_eye
from config import parameters

#class to model a lane_line
class Line:

    
    def __init__(self, buffer_len= 10):
        
        #this flag marks whether the line was detected the last iteration
        self.detected= False

        #polynomial coeffs fitted on the last iteration
        self.last_fit_pixel= None
        self.last_fit_meter= None 

        #list of polynomial coeffs of the last N iterations
        self.recent_fits_pixel= collections.deque(maxlen= buffer_len)
        self.recent_fits_meter= collections.deque(maxlen= 2* buffer_len)

        self.radius_of_curvature= None

        #all pixels (x, y) of line detected
        self.all_x= None
        self.all_y= None 

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer= False):
        '''
        update line with new fitted coeffs

        paramters:
        _______________
        new_fit_pixel: new polynomial coeffs (pixel)
        new_fit_meter: new polynomial coeffs (meter)
        detected: if the line was detected or inferred
        clear_buffer: if True, reset state
        '''

        self.detected= detected 

        if clear_buffer:
            self.recent_fits_meter= []
            self.recent_fits_pixel= []

        self.last_fit_pixel= new_fit_pixel 
        self.last_fit_meter= new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    
    def draw(self, mask, color= (255, 0, 0), line_width=50, average= False):
        '''
        draw the line on a color mask image
        '''
        h, w, c= mask.shape

        
        coeffs= self.average_fit if average else self.last_fit_pixel

        '''
        plot_y here is like the x axis that any function is evaluated w.r.t. but here we use y because the 
        function we're representing (the lane line) is parallel to y (height) and perpendicular to x (width)
        '''
        plot_y= np.linspace(0, h-1, h)
        line_center= (coeffs[0]*(plot_y**2)) + (coeffs[1]*plot_y) + coeffs[2]
        line_left_side= line_center- (line_width//2)
        line_right_side= line_center+ (line_width//2)

        #change x and y points to a form that can be used in cv2.fillpoly()
        pts_left= np.array(list(zip(line_left_side, plot_y)))
        pts_right= np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts= np.vstack([pts_left, pts_right])

        return cv2.fillPoly(mask, [np.int32(pts)], color)
    
    def print_coeffs(self):
        coeffs= self.last_fit_pixel
            
        for coeff in coeffs:
            print(coeff, ", ")
        
    @property
    def average_fit(self): #average of polynomial coeffs in the last N iterations
        return np.mean(self.recent_fits_pixel, axis= 0)
    
    @property
    def curvature(self): #radius of curvature of the line (averaged)
        y_eval= 0
        coeffs= self.average_fit
        
        return ((1 + (2* coeffs[0] * y_eval+ coeffs[1])**2) ** 1.5) / np.absolute(2 * coeffs[0])

    
    @property
    def curvature_meter(self):
        y_eval= 0
        coeffs= np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2* coeffs[0] * y_eval+ coeffs[1])**2) ** 1.5) / np.absolute(2 * coeffs[0])
