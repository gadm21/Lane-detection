import cv2
import numpy as np

image= cv2.imread("road2.PNG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_h, image_w= image.shape[:2]


histogram = np.sum(gray[image_h//2: image_h-30], axis=0)


midpoint= len(histogram)//2
peak_left= np.argmax(histogram[:midpoint])
peak_right= np.argmax(histogram[midpoint:])+ midpoint


window_num= 30
window_h= image_h/ window_num

nonzero= gray.nonzero()
nonzero_x= np.array(nonzero[1])
nonzero_y= np.array(nonzero[0])

current_left= peak_left
current_right=peak_right

margin= 50
minpix= 8


for window in range(window_num):
    
    win_y_low= int(image_h - (1+ window)* window_h)
    win_y_high= int(image_h - (window)* window_h)
    
    win_x_low_left= int(current_left- margin)
    win_x_high_left= int(current_left+ margin)
    
    win_x_low_right= int(current_right- margin)
    win_x_high_right= int(current_right+ margin)

    cv2.rectangle(image, (win_x_low_right, win_y_low), (win_x_high_right, win_y_high),  (0, 255, 0), 2)
    cv2.rectangle(image, (win_x_low_left, win_y_low), (win_x_high_left, win_y_high), (0, 255, 0), 2)


    nonzero_inside_left_window= ((nonzero_y > win_y_low) & (nonzero_y < win_y_high) &
                                   (nonzero_x > win_x_low_left) & (nonzero_x < win_x_high_left)).nonzero()[0]
    nonzero_inside_right_window= ((nonzero_y > win_y_low) & (nonzero_y < win_y_high) &
                                   (nonzero_x > win_x_low_right) & (nonzero_x < win_x_high_right)).nonzero()[0]                      

    if len(nonzero_inside_left_window) > minpix :
        current_left= np.int(np.mean(nonzero_x[nonzero_inside_left_window]))
        
    if len(nonzero_inside_right_window) > minpix :
        current_right= np.int(np.mean(nonzero_x[nonzero_inside_right_window]))
    
    cv2.circle(image, (current_left, (win_y_low+ win_y_high)//2), 5, (0, 0, 255))
    cv2.circle(image, (current_right, (win_y_low+ win_y_high)//2), 5, (0, 0, 255))
        

cv2.circle(image, (midpoint, image_h-30), 5, (255, 0, 0))

cv2.imshow("R", image)
cv2.waitKey(0)
cv2.destroyWindow("R")
