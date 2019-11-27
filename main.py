from config import parameters
from camera_generals import cam_cal
from moviepy.editor import VideoFileClip
from general_helper import lane_finding_pipeline
from Line import Line
import os
import cv2

left_line= Line() 
right_line= Line() 
processed_frames= 0 

if __name__ == "__main__":

    # calibrate the camera
    found , cam_mtx, dist_coeffs, rotation_v, translation_v= cam_cal(parameters["cal_img_dir"])
    parameters["cam_mtx"]= cam_mtx
    parameters["dist_coeffs"]= dist_coeffs
    parameters["r_v"]= rotation_v
    parameters["t_v"]= translation_v
    
    if parameters["mode"] == "video":
        clip= VideoFileClip(parameters["test_video"])
        output_clip= clip.fl_image(lane_finding_pipeline)
        output_clip.write_videofile(parameters["output_video"], audio= False)
    else:
        test_images_dir= parameters["test_image_dir"]
        for test_image_file in os.listdir(test_images_dir):
            image= cv2.imread(os.path.join(test_images_dir, test_image_file))
            output_image= lane_finding_pipeline(image, left_line, right_line, processed_frames)
            
            cv2.imshow("r", output_image)
            cv2.waitKey(0)
            cv2.destroyWindow("r")
            
            cv2.imwrite(os.path.join(parameters["output_image_dir"], test_image_file), output_image)
            
            
            
    