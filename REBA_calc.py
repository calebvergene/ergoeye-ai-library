import cv2
import numpy as np
import time
from pose_module import poseDetector

"""
- need to calculate all REBA steps per frame

- create a class that takes in (image,landmark_list) as input, then runs all calculations for REBA steps with each body test
- then, calculate the final score and display the score for each frame
"""

class calcREBAPose():
    def __init__(self, img, landmark_list):
        self.img = img
        self.landmark_list = landmark_list

    def calc_upper_arm(self, angle):
        score = 0
        if angle <= 20 or angle >= 340:
            score = 1
        elif angle >= 260:
            score = 2
        elif angle > 20 and angle <= 45:
            score = 2
        elif angle > 45 and angle <= 90:
            score = 3
        else: score = 4
        print('Upper Arm Angle:', angle)
        print('Upper Arm Score:', score)

        ## TODO: need to code a way to choose the angle of the side of your body that has the highest confidence. 
        ## for example, if it is easier to detect your right arm, then the code should calculate the right side of 
        ## your body. THEN, input either "right" or "left" into the function so calculations can be consistent. 




