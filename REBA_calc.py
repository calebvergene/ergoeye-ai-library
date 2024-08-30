import math
import cv2


"""
- need to calculate all REBA steps per frame

- create a class that takes in (image,landmark_list) as input, then runs all calculations for REBA steps with each body test
- then, calculate the final score and display the score for each frame
"""

def calc_neck(direction, nose, shoulder, hip, img, pose_detector):

    left_shoulder = shoulder[0]
    right_shoulder = shoulder[1]


    shoulder_midpoint_x = (left_shoulder['x'] + right_shoulder['x']) / 2
    shoulder_midpoint_y = (left_shoulder['y'] + right_shoulder['y']) / 2
    shoulder_midpoint_dict = {'x': shoulder_midpoint_x, 'y':shoulder_midpoint_y}
    cv2.circle(img, (int(shoulder_midpoint_x),int(shoulder_midpoint_y)), 5, (0,255,0), cv2.FILLED)
    hip_midpoint_x = (hip[0]['x'] + hip[1]['x']) / 2
    hip_midpoint_y = (hip[0]['y'] + hip[1]['y']) / 2
    hip_midpoint_dict = {'x': hip_midpoint_x, 'y':hip_midpoint_y}
    cv2.circle(img, (int(hip_midpoint_x),int(hip_midpoint_y)), 5, (0,255,0), cv2.FILLED)
    # Calculate the angle between the neck line and the vertical axis

    neck_angle = pose_detector.find_angle(img, hip_midpoint_dict, shoulder_midpoint_dict, nose) - 180
    if direction == 'right':
        neck_angle -=30
    else:
        neck_angle +=30
    

    print(f'neck angle: {neck_angle}')




def calc_upper_arm(angle):
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


def execute_REBA_test(pose_detector, img):
    landmark_list = pose_detector.find_position(img)
    neck_direction = pose_detector.find_direction([7], [8]) #based off ear
    neck_result = calc_neck(neck_direction, landmark_list[0], [landmark_list[11], landmark_list[12]], [landmark_list[23], landmark_list[24]], img, pose_detector)
    print(neck_result)
