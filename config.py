import numpy as np

parameters= {
    "cal_img_dir": "camera_cal",
    "test_image_dir": "test_images",
    "output_image_dir":"output_images",
    "mode": "image",
    "dist_coeffs": None,
    "cam_mtx": None,
    "yellow_thresh_min": np.array([0, 70, 70]),
    "yellow_thresh_max": np.array([50, 255, 255]),
    "sobel_kernel": 9,
    "window_num": 9,
    "window_width":200,
    "min_pix_recenter":50,
    "met_per_pix": 10,
     }