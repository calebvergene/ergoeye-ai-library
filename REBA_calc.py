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
        # To accomodate for the person facing the opposite direction
        neck_angle -=90
        neck_angle = neck_angle * -1
    print(f'neck angle: {neck_angle}')

    # Calculate REBA score
    if neck_angle >= 20:
        return 2
    elif neck_angle <= 0:
        return 2
    else: 
        return 1
    

def calc_trunk(direction, shoulder, hip, img, pose_detector):
    """
    Finds angle between middle should, middle hip, and below middle hip (-180 degrees for accurate trunk tilt)

    Returns a score based off of the REBA trunk test 

    NEED TO ADD: Neck side bending and neck twist
    NEED TESTING
    FOR THIS TO BE ACCURATE: Camera needs to be centered so that gravity is going directly down. 3rd point is directly below hip
    """

    left_shoulder = shoulder[0]
    right_shoulder = shoulder[1]

    # Calculate middle points of shoulder and hip
    shoulder_midpoint_x = (left_shoulder['x'] + right_shoulder['x']) / 2
    shoulder_midpoint_y = (left_shoulder['y'] + right_shoulder['y']) / 2
    shoulder_midpoint_dict = {'x': shoulder_midpoint_x, 'y':shoulder_midpoint_y}
    ### cv2.circle(img, (int(shoulder_midpoint_x),int(shoulder_midpoint_y)), 5, (0,255,0), cv2.FILLED)
    hip_midpoint_x = (hip[0]['x'] + hip[1]['x']) / 2
    hip_midpoint_y = (hip[0]['y'] + hip[1]['y']) / 2
    hip_midpoint_dict = {'x': hip_midpoint_x, 'y':hip_midpoint_y}
    ### cv2.circle(img, (int(hip_midpoint_x),int(hip_midpoint_y)), 5, (0,255,0), cv2.FILLED)
    beneath_hip_point_dict = {'x': hip_midpoint_x, 'y': hip_midpoint_y + 50}  # 50 pixels beneath
    ### cv2.circle(img, (int(beneath_hip_point_dict['x']), int(beneath_hip_point_dict['y'])), 5, (0, 255, 0), cv2.FILLED)

    # Modify angle to be accurate.
    trunk_angle = pose_detector.find_angle(img, beneath_hip_point_dict, hip_midpoint_dict, shoulder_midpoint_dict) - 180
    if direction == 'left':
        trunk_angle = trunk_angle * -1
    print(f'trunk angle: {trunk_angle}')

    # Calculate REBA score
    if trunk_angle >= 60:
        return 4
    elif trunk_angle <= -20:
        return 3
    elif trunk_angle >= 20:
        return 3
    elif trunk_angle >= 0:
        return 2
    elif trunk_angle <= 0:
        return 2
    else: 
        return 1
    

def calc_legs(direction, hip, knee, ankle, img, pose_detector):
    """
    Finds angle between hip, knee, and ankle

    Returns a score based off of the REBA trunk test 

    NEED TO ADD:
    NEED TESTING
    """

    left_hip = hip[0]
    right_hip = hip[1]

    left_knee = knee[0]
    right_knee = knee[1]

    left_ankle = ankle[0]
    right_ankle = ankle[1]

    # Modify angle to be accurate.
    left_leg_angle = pose_detector.find_angle(img, left_hip, left_knee, left_ankle) - 180
    right_leg_angle = pose_detector.find_angle(img, right_hip, right_knee, right_ankle) - 180

      
    if direction == 'left':
        left_leg_angle = left_leg_angle * -1
        right_leg_angle = right_leg_angle * -1
    

    print(f'left leg angle: {left_leg_angle}')
    print(f'right leg angle: {right_leg_angle}')

    if left_leg_angle > right_leg_angle:
        leg_angle = right_leg_angle
    else:
        leg_angle = left_leg_angle

    # Calculate REBA score
    if leg_angle >= 60:
        return 3
    elif leg_angle >= 30:
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
    direction = pose_detector.find_direction(landmark_list) #based off ear

    neck_result = calc_neck(direction, landmark_list[0], [landmark_list[11], landmark_list[12]], [landmark_list[7], landmark_list[8]], img, pose_detector)
    print(f'neck score: {neck_result}')

    trunk_result = calc_trunk(direction, [landmark_list[11], landmark_list[12]], [landmark_list[23], landmark_list[24]], img, pose_detector)
    print(f'trunk score: {trunk_result}')

    leg_result = calc_legs(direction, [landmark_list[23], landmark_list[24]], [landmark_list[25], landmark_list[26]], [landmark_list[27], landmark_list[28]], img, pose_detector)
    print(f'leg score: {leg_result}')