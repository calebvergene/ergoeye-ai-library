import cv2
import numpy as np
import time
import pose_module as pm 

"""
- need to calculate all RULA steps per frame

- create a class that takes in (image,landmark_list) as input, then runs all calculations for RULA steps with each body test
- then, each method returns a number. with each frame being passed through this class, we keep the highest number over the course of the video
- finally, calculate the final score with another function 
"""

