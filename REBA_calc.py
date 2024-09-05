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
    Finds angle between middle shoulder, middle hip, and below middle hip (-180 degrees for accurate trunk tilt)

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

    NEED TO ADD: if leg is lifted???
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


def first_REBA_score(neck_score, trunk_score, leg_score):
    # Define the posture score table for each neck score
    posture_score_table = {
        1: [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [2, 4, 5, 6],
            [3, 5, 6, 7],
            [4, 6, 7, 8]
        ],
        2: [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8]
        ],
        3: [
            [3, 4, 5, 6],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9]
        ]
    }
    
    # Subtract 1 from the scores to match the table indices (0-indexed in Python)
    trunk_index = trunk_score - 1
    leg_index = leg_score - 1
    
    posture_score = posture_score_table[neck_score][trunk_index][leg_index]
    
    return posture_score


def calc_upper_arm(direction, hip, shoulder, elbow, img, pose_detector):
    """
    Finds angle between hip, shoulder, and elbow to find upper arm angle

    Returns a score based off of the REBA upper arm test 

    NEED TO ADD: if shoulder raised, upper arm abducted, arm supported/person leaning
    NEED TESTING
    """

    left_hip = hip[0]
    right_hip = hip[1]
    left_shoulder = shoulder[0]
    right_shoulder = shoulder[1]
    left_elbow = elbow[0]
    right_elbow = elbow[1]

    # Find angle
    left_upper_arm_angle = pose_detector.find_angle(img, left_hip, left_shoulder, left_elbow)
    right_upper_arm_angle = pose_detector.find_angle(img, right_hip, right_shoulder, right_elbow)
    
    # Modify angle to be accurate
    # Because humanly impossible to have your arms behind your head at at 150 degree angle, 
    # this calculation flips the angle to adjust to the correct calculation
    if direction == 'right':
        if left_upper_arm_angle >= 150:
            left_upper_arm_angle = left_upper_arm_angle - 360 
        if right_upper_arm_angle >= 150:
            right_upper_arm_angle = right_upper_arm_angle - 360
        left_upper_arm_angle = left_upper_arm_angle * -1
        right_upper_arm_angle = right_upper_arm_angle * -1
    elif direction == 'left':
        if left_upper_arm_angle >= 210:
            left_upper_arm_angle = left_upper_arm_angle - 360 
        if right_upper_arm_angle >= 210:
            right_upper_arm_angle = right_upper_arm_angle - 360


    print(f'left upper arm angle: {left_upper_arm_angle}')
    print(f'right upper arm angle: {right_upper_arm_angle}')

    if left_upper_arm_angle > right_upper_arm_angle:
        upper_arm_angle = right_upper_arm_angle
    else:
        upper_arm_angle = left_upper_arm_angle

    # Calculate REBA score
    if upper_arm_angle >= 90:
        return 4
    elif upper_arm_angle >= 45:
        return 3
    elif upper_arm_angle >= 20:
        return 2
    elif upper_arm_angle <= 20:
        return 2
    else: 
        return 1
    

def calc_upper_arm(direction, hip, shoulder, elbow, img, pose_detector):
    """
    Finds angle between hip, shoulder, and elbow to find upper arm angle

    Returns a score based off of the REBA upper arm test 

    NEED TO ADD: if shoulder raised, upper arm abducted, arm supported/person leaning
    NEED TESTING
    """

    left_hip = hip[0]
    right_hip = hip[1]
    left_shoulder = shoulder[0]
    right_shoulder = shoulder[1]
    left_elbow = elbow[0]
    right_elbow = elbow[1]

    # Find angle
    left_upper_arm_angle = pose_detector.find_angle(img, left_hip, left_shoulder, left_elbow)
    right_upper_arm_angle = pose_detector.find_angle(img, right_hip, right_shoulder, right_elbow)
    
    # Modify angle to be accurate
    # Because humanly impossible to have your arms behind your head at at 150 degree angle, 
    # this calculation flips the angle to adjust to the correct calculation
    if direction == 'right':
        if left_upper_arm_angle >= 150:
            left_upper_arm_angle = left_upper_arm_angle - 360 
        if right_upper_arm_angle >= 150:
            right_upper_arm_angle = right_upper_arm_angle - 360
        left_upper_arm_angle = left_upper_arm_angle * -1
        right_upper_arm_angle = right_upper_arm_angle * -1
    elif direction == 'left':
        if left_upper_arm_angle >= 210:
            left_upper_arm_angle = left_upper_arm_angle - 360 
        if right_upper_arm_angle >= 210:
            right_upper_arm_angle = right_upper_arm_angle - 360


    print(f'left upper arm angle: {left_upper_arm_angle}')
    print(f'right upper arm angle: {right_upper_arm_angle}')

    if left_upper_arm_angle > right_upper_arm_angle:
        upper_arm_angle = right_upper_arm_angle
    else:
        upper_arm_angle = left_upper_arm_angle

    # Calculate REBA score
    if upper_arm_angle >= 90:
        return 4
    elif upper_arm_angle >= 45:
        return 3
    elif upper_arm_angle >= 20:
        return 2
    elif upper_arm_angle <= 20:
        return 2
    else: 
        return 1
    

def calc_lower_arm(direction, wrist, shoulder, elbow, img, pose_detector):
    """
    Finds angle between wrist, shoulder, and elbow to find lower arm angle

    Returns a score based off of the REBA lower arm test 

    NEED TESTING
    """

    left_wrist = wrist[0]
    right_wrist = wrist[1]
    left_shoulder = shoulder[0]
    right_shoulder = shoulder[1]
    left_elbow = elbow[0]
    right_elbow = elbow[1]

    # Find angle
    left_lower_arm_angle = pose_detector.find_angle(img, left_shoulder, left_elbow, left_wrist)
    right_lower_arm_angle = pose_detector.find_angle(img, right_shoulder, right_elbow, right_wrist)
 

    if direction == 'left':
        left_lower_arm_angle =  360 - left_lower_arm_angle
        right_lower_arm_angle = 360 - right_lower_arm_angle


    print(f'left lower arm angle: {left_lower_arm_angle}')
    print(f'right lower arm angle: {right_lower_arm_angle}')

    if left_lower_arm_angle > right_lower_arm_angle:
        lower_arm_angle = right_lower_arm_angle
    else:
        lower_arm_angle = left_lower_arm_angle

    # Calculate REBA score
    if lower_arm_angle <= 80:
        return 2
    elif lower_arm_angle >= 120:
        return 2
    else: 
        return 1



def execute_REBA_test(pose_detector, img):
    landmark_list = pose_detector.find_position(img)
    direction = pose_detector.find_direction(landmark_list) #based off ear

    neck_result = calc_neck(direction, landmark_list[0], [landmark_list[11], landmark_list[12]], [landmark_list[7], landmark_list[8]], img, pose_detector)
    print(f'neck score: {neck_result}')

    trunk_result = calc_trunk(direction, [landmark_list[11], landmark_list[12]], [landmark_list[23], landmark_list[24]], img, pose_detector)
    print(f'trunk score: {trunk_result}')

    leg_result = calc_legs(direction, [landmark_list[23], landmark_list[24]], [landmark_list[25], landmark_list[26]], [landmark_list[27], landmark_list[28]], img, pose_detector)
    print(f'leg score: {leg_result}')

    # Need to implement step 5; weight of object

    reba_score_1 = first_REBA_score(neck_result, trunk_result, leg_result)

    upper_arm_result = calc_upper_arm(direction, [landmark_list[23], landmark_list[24]], [landmark_list[11], landmark_list[12]], [landmark_list[13], landmark_list[14]], img, pose_detector)
    print(f'upper arm score: {upper_arm_result}')

    lower_arm_result = calc_lower_arm(direction, [landmark_list[15], landmark_list[16]], [landmark_list[11], landmark_list[12]], [landmark_list[13], landmark_list[14]], img, pose_detector)
    print(f'lower arm score: {lower_arm_result}')