from config import parameters
from camera_generals import undistort, binarize, bird_eye
import numpy as np
import cv2


def draw_lanes_on_road(undistorted_image, inv_mtx, left_line, right_line, keep_state):
    '''draw both the drivable lane area and the detected
    lane lines onto the original frame.
    parameters:
    ____________
    undistorted_image: original undistorted color frame
    inv_mtx: (inverse) perspective transform mtx used to
        reproject on original frame
    left_line: left lane line previously detected
    right_line:right lane line previously detected
    keepstate: if True, line state is maintained
    returns:
    _________
    color blend
    '''

    #extract dimensions of the frame
    height, width, _= undistorted_image.shape

    #extract polynomial coefficients from each lane line
    left_coeffs= left_line.average_fit if keep_state else left_line.last_fit_pixel
    right_coeffs= right_line.average_fit if keep_state else right_line.last_fit_pixel

    #generate x and y values for plotting
    y= np.linspace(0, height-1, height)
    left_x= left_coeffs[0]* (y**2) + left_coeffs[1]* y + left_coeffs[2]
    right_x= right_coeffs[0]* (y**2)+ right_coeffs[1]*y+ right_coeffs[2]

    #draw green polygon on drivable area (between the two lanes)
    road_wrap= np.zeros_like(undistorted_image, dtype= np.uint8)
    left_pts= np.array([np.transpose(np.vstack([left_x, y]))])
    right_pts= np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
    pts= np.hstack((left_pts, right_pts))
    cv2.fillPoly(road_wrap, np.int_([pts]), (0, 255, 0))
    # convert the mask with the drawn polygon on it from birdeye
    # to the original frame prespective using the inverse matrix
    road_dewarped= cv2.warpPerspective(road_wrap, inv_mtx, (width, height))
    # put the road polygon on the original frame
    blend_onto_road= cv2.addWeighted(undistorted_image, 1, road_dewarped, 0.3, 0)

#    cv2.imshow("blend_onto_road", road_wrap)
#    cv2.waitKey(0)
#    cv2.destroyWindow("blend_onto_road")
    
    #draw solid line on each lane (on bideye perspective)
    line_wrap= np.zeros_like(undistorted_image)
    line_wrap= left_line.draw(line_wrap, color= (255, 0, 0), average= keep_state)
    line_wrap= right_line.draw(line_wrap, color= (0, 0, 255), average= keep_state)
    #change perspective to original frame
    line_dewarped= cv2.warpPerspective(line_wrap, inv_mtx, (width, height))

    #why not simply addWeight of hte lane lines to the road image ?

    lines_mask= blend_onto_road.copy()
    idx= np.any([line_dewarped!=0][0], axis=2)
    lines_mask[idx]= line_dewarped[idx]

    #try delete the following line and just return lines_mask
    blend_onto_road= cv2.addWeighted(lines_mask, 0.8, blend_onto_road, 0.5, 0)


    return blend_onto_road


def compute_offset_from_center(left_line, right_line, width, parameters):
    '''
    compute offset from the center of the inferred lane.
    The offset from the lane center can be computed under
    the hypothesis that the camera is fixed and mounted
    in the midpoint of the car roof.
    
    returns:
    ________
    offset
    
    '''
    
    if left_line.detected and right_line.detected:
        left_lane_x= np.mean(left_line.all_x[left_line.all_y > (0.95 * left_line.all_y.max())])
        right_lane_x= np.mean(right_line.all_x[right_line.all_y > (0.95 * right_line.all_y.max())])
        distance_between_lanes= right_lane_x- left_lane_x
        
        frame_midpoint= width//2
        lanes_midpoint= left_lane_x+ (distance_between_lanes//2)

        offset_pix= abs(lanes_midpoint- frame_midpoint)
        offset_meter= parameters["met_per_pix"] * offset_pix

    else: offset_meter= -1

    return offset_meter

def get_fits_by_previous_fits(birdeye_image, left_line, right_line, parameters):
    '''
    get polynomial coeffs for previously detected lane-lines in a binary image
    This function starts from a previously detected lane_lines to speed-up searching

    returns:
    ___________
    updated lane lines and output_image
    '''
    

    '''
    get the height and width of the birdeye image
    why birdeye_image has only 2 dimensions?
    '''
    image_h, image_w= birdeye_image.shape

    '''
    get the coeffs of the lanes that was previous extracted
    '''
    left_lane_coeffs= left_line.last_fit_pixel
    right_lane_coeffs= right_line.last_fit_pixel

    '''
    Find the nonzero pixels in the whole image.
    Define the width of the lane (margin).
    Find the pixels that exits within the lane boundaries
    '''
    nonzero = birdeye_image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100

    left_lane_inds = (
    (nonzero_x > (left_lane_coeffs[0] * (nonzero_y ** 2) + left_lane_coeffs[1] * nonzero_y + left_lane_coeffs[2] - margin)) & (
    nonzero_x < (left_lane_coeffs[0] * (nonzero_y ** 2) + left_lane_coeffs[1] * nonzero_y + left_lane_coeffs[2] + margin)))
    
    right_lane_inds = (
    (nonzero_x > (right_lane_coeffs[0] * (nonzero_y ** 2) + right_lane_coeffs[1] * nonzero_y + right_lane_coeffs[2] - margin)) & (
    nonzero_x < (right_lane_coeffs[0] * (nonzero_y ** 2) + right_lane_coeffs[1] * nonzero_y + right_lane_coeffs[2] + margin)))

    # Extract left and right line pixel positions
    left_line.all_x, left_line.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    right_line.all_x, right_line.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
    
    '''
    If you didn't find any pixels within the lane boundaries,
    set the coeffs to the last coeffs found. 
    Else, fit the y and x values of the found pixels into a new polynomial.
    Update the line with the coeffs
    '''
    detected= True
    if not list(left_line.all_x) or not list(left_line.all_y):
        left_fit_pixel = left_line.last_fit_pixel
        left_fit_meter = left_line.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(left_line.all_y, left_line.all_x, 2)
        left_fit_meter = np.polyfit(left_line.all_y * parameters["met_per_pix"], left_line.all_x * parameters["met_per_pix"], 2)

    if not list(right_line.all_x) or not list(right_line.all_y):
        right_fit_pixel = right_line.last_fit_pixel
        right_fit_meter = right_line.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(right_line.all_y, right_line.all_x, 2)
        right_fit_meter = np.polyfit(right_line.all_y * parameters["met_per_pix"], right_line.all_x * parameters["met_per_pix"], 2)

    left_line.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    right_line.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    '''
    Using coeffs, generate x and y values for left and right lanes to plot
    '''
    y= np.linspace(0, image_h-1, image_h)
    left_x= left_fit_pixel[0]* (y**2) + left_fit_pixel[1]* y + left_fit_pixel[2]
    right_x= right_fit_pixel[0]* (y**2)+ right_fit_pixel[1]*y+ right_fit_pixel[2]

    img_fit= np.dstack((birdeye_image, birdeye_image, birdeye_image))
    window_img= np.zeros_like(img_fit)
    
    #color left and right lanes
    img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]]= [255, 0, 0]
    img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]]= [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_x - margin, y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_x + margin, y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_x - margin, y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_x + margin, y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    #draw the lanes on the mask
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result= cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

    return left_line, right_line, img_fit




def get_fit_by_sliding_windows(birdeye_binary, line_lt, line_rt, parameters):
    """
    Get polynomial coefficients for lane-lines detected in an binary image.

    :param birdeye_binary: input bird's eye view binary image
    :param line_lt: left lane-line previously detected
    :param line_rt: left lane-line previously detected
    :param n_windows: number of sliding windows used to search for the lines
    :param verbose: if True, display intermediate output
    :return: updated lane lines and output image
    """
    height, width = birdeye_binary.shape
    
    n_windows=14
    ym_per_pix= parameters["met_per_pix"]
    xm_per_pix= ym_per_pix
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(birdeye_binary[height//2:-30, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(height / n_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100  # width of the windows +/- margin
    minpix = 50   # minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    print("................................")
    line_lt.print_coeffs()
    line_rt.print_coeffs()
    print("................................")
    
    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]


    return line_lt, line_rt, out_img



def get_fit_by_sliding_windows2(birdeye_image, left_line, right_line, parameters):

    

    
    '''
    get the height and width of the birdeye image
    why birdeye_image has only 2 dimensions?
    '''
    image_h, image_w= birdeye_image.shape

    '''
    histogram= sum the pexels vertically from h//2 (floor division)
    to h-30
    
    find the index (x-index) of the peak value from the 
    left and right halves of the histogram
    '''
    histogram= np.sum(birdeye_image[image_h//2:-30, :], axis=0)
    midpoint= len(histogram)//2
    left_peak_index= np.argmax(histogram[:midpoint])
    right_peak_index= np.argmax(histogram[midpoint:]) + midpoint

    '''
    define the window height, it's the image_height/ window_num
    we set window_num= 10
    '''
    window_num= parameters["window_num"]
    window_h= np.int(image_h/window_num)

    '''
    identify the indecies of the nonzero pexels
    and keep track of the x and y values separately
    '''
    nonzero= birdeye_image.nonzero()
    nonzero_y= np.array(nonzero[0])
    nonzero_x= np.array(nonzero[1])

    '''
    identify:
    1- the center of the first window (peaks)
    2- width of the window
    3- min_zeros: minimum number of nonzero pexels inside a window 
        to recenter the next window
    '''
    left_window_center= left_peak_index
    right_window_center= right_peak_index

    window_width= parameters["window_width"]
    min_pix= parameters["min_pix_recenter"]

    all_left_lane_indicies= []
    all_right_lane_indicies= []

    '''
    for each step (you have 2 windows at each step, right & left):
        1- identify the boundaries for the 2 windows
        2-identify the indecies from the nonzeros lists that exists
            inside the windows boundaires
        3- if len(nonzero_pexels) > min_zeros: recenter the next window
            by taking the average of the current nonzero_x indecies
    '''

    #create an output image to draw lanes on
    out_image= np.dstack((birdeye_image, birdeye_image, birdeye_image))* 255

    for window in range(window_num):
        #boundaries
        win_y_low= image_h- ((window+1)*window_h)
        win_y_high= image_h- (window* window_h)
        win_x_left_low= left_window_center- (window_width//2)
        win_x_left_high= left_window_center+ (window_width//2)
        win_x_right_low= right_window_center- (window_width//2)
        win_x_right_high= right_window_center+ (window_width//2)

        #visulaize 
        cv2.rectangle(out_image, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_image, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)

        nonzero_left_window= ((nonzero_y > win_y_low) &
                             (nonzero_y < win_y_high) &
                             (nonzero_x > win_x_left_low) &
                             (nonzero_x < win_x_left_high)).nonzero()[0]
        nonzero_right_window= ((nonzero_y > win_y_low) &
                               (nonzero_y < win_y_high)&
                               (nonzero_x > win_x_right_low)&
                               (nonzero_x < win_x_right_high)).nonzero()[0]
        
        all_left_lane_indicies.append(nonzero_left_window)
        all_right_lane_indicies.append(nonzero_right_window)

        if len(nonzero_left_window) > min_pix:
            left_window_center= np.int(np.mean(nonzero_x[nonzero_left_window]))
        if len(nonzero_right_window) > min_pix:   
            right_window_center= np.int(np.mean(nonzero_x[nonzero_right_window]))
    

    '''
    gather all the nonzero indicies that was found insides any of the 
    windows, return their corresponding (x, y) pexels to line.all_x and line.all_y
    '''
    all_left_lane_indicies= np.concatenate(all_left_lane_indicies)
    all_right_lane_indicies= np.concatenate(all_right_lane_indicies)

    left_line.all_x, left_line.all_y= nonzero_x[all_left_lane_indicies], nonzero_y[all_left_lane_indicies]
    right_line.all_x, right_line.all_y= nonzero_x[all_right_lane_indicies], nonzero_y[all_right_lane_indicies]

    '''
    fit the found pixels in a polynomial trajectory that defines the line
    If you didn't any pixels in this frame to fit, use previous frames pixels.
    '''
    ym_per_pix= parameters["met_per_pix"]
    xm_per_pix= ym_per_pix
    detected = True
    if not list(left_line.all_x) or not list(left_line.all_y):
        left_fit_pixel = left_line.last_fit_pixel
        left_fit_meter = left_line.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(left_line.all_y, left_line.all_x, 2)
        left_fit_meter = np.polyfit(left_line.all_y * ym_per_pix, left_line.all_x * xm_per_pix, 2)

    if not list(right_line.all_x) or not list(right_line.all_y):
        right_fit_pixel = right_line.last_fit_pixel
        right_fit_meter = right_line.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(right_line.all_y, right_line.all_x, 2)
        right_fit_meter = np.polyfit(right_line.all_y * ym_per_pix, right_line.all_x * xm_per_pix, 2)


    left_line.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    right_line.update_line(right_fit_pixel, right_fit_meter, detected=detected)


    out_image[nonzero_y[all_left_lane_indicies], nonzero_x[all_left_lane_indicies]]= [255, 0, 0]
    out_image[nonzero_y[all_right_lane_indicies], nonzero_x[all_right_lane_indicies]]= [0, 0, 255]

#    cv2.imshow("out", out_image)
#    cv2.waitKey(0)
#    cv2.destroyWindow("out")
    
    return left_line, right_line, out_image



def lane_finding_pipeline(image, left_line, right_line, processed_frames, keep_state= False):
    
   # global left_line, right_line, processed_frames
    #undistored the image (assuming camera is calibrated)
    undistorted_image= undistort(image)

    #binarize the image to highlight lines
    binary_image= binarize(undistorted_image)

    #compute birdeye view
    birdeye_image, matrix, matrix_inv= bird_eye(binary_image)
    


    if processed_frames>0 and keep_state and left_line.detected and right_line.detected:
        left_line, right_line, image_fit= get_fits_by_previous_fits(birdeye_image, left_line, right_line, parameters)
    else:
        left_line, right_line, image_fit= get_fit_by_sliding_windows2(birdeye_image, left_line, right_line, parameters)
     

    #compute offset from the center (in meters)
    #offset_meters= compute_offset_from_center(left_line, right_line, image.shape[1], parameters)

    #draw lane lines back on the original image
    blend_on_road= draw_lanes_on_road(undistorted_image, matrix_inv, left_line, right_line, keep_state)

    processed_frames+=1

    return blend_on_road 