import math
import cv2


"""
- need to calculate all REBA steps per frame

- create a class that takes in (image,landmark_list) as input, then runs all calculations for REBA steps with each body test
- then, calculate the final score and display the score for each frame
"""

def calc_neck(direction, nose, shoulder, ear, img, pose_detector):
    """
    Finds angle between middle shoulder, middle ear, and nose (-90 degrees for neck angle)

    Returns a score based off of the REBA neck test 

    NEED TO ADD: Neck bending and neck twist
    NEED TESTING
    """

    left_shoulder = shoulder[0]
    right_shoulder = shoulder[1]

    # Calculate middle points of shoulder and ear
    shoulder_midpoint_x = (left_shoulder['x'] + right_shoulder['x']) / 2
    shoulder_midpoint_y = (left_shoulder['y'] + right_shoulder['y']) / 2
    shoulder_midpoint_dict = {'x': shoulder_midpoint_x, 'y':shoulder_midpoint_y}
    ### cv2.circle(img, (int(shoulder_midpoint_x),int(shoulder_midpoint_y)), 5, (0,255,0), cv2.FILLED)
    ear_midpoint_x = (ear[0]['x'] + ear[1]['x']) / 2
    ear_midpoint_y = (ear[0]['y'] + ear[1]['y']) / 2
    ear_midpoint_dict = {'x': ear_midpoint_x, 'y':ear_midpoint_y}
    ### cv2.circle(img, (int(ear_midpoint_x),int(ear_midpoint_y)), 5, (0,255,0), cv2.FILLED)

    # Modify angle to be accurate.
    neck_angle = pose_detector.find_angle(img, shoulder_midpoint_dict, ear_midpoint_dict, nose) 
    if direction == 'right':
        neck_angle -=270
    else:
        neck_angle -=90
        neck_angle = neck_angle * -1
    print(f'neck angle: {neck_angle}')

    # Calculate REBA score
    if neck_angle >= 15:
        return 2
    elif neck_angle <= 5:
        return 2
    else: 
        return 1


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
    neck_direction = pose_detector.find_direction(landmark_list) #based off ear
    neck_result = calc_neck(neck_direction, landmark_list[0], [landmark_list[11], landmark_list[12]], [landmark_list[7], landmark_list[8]], img, pose_detector)
    print(neck_result)
