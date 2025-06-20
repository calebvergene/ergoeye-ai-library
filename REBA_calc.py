import numpy as np
import cv2



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
    ###print(f'neck angle: {neck_angle}')

    

    # Calculate REBA score
    if neck_angle >= 20:
        critical_limbs.append({"neck": neck_angle})
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
    ###print(f'trunk angle: {trunk_angle}')


    # Draws the line for trunk
    def trunk_color(img, angle, p1, p2):
        p1_coords = (int(p1['x']), int(p1['y']))
        p2_coords = (int(p2['x']), int(p2['y']))
        if angle >= 60:
            cv2.line(img, p1_coords, p2_coords, (61, 61, 255), 11)  # Red
        elif angle >= 20 or angle <= -20:
            cv2.line(img, p1_coords, p2_coords, (42, 212, 227), 11)  # Yellow
        else:
            cv2.line(img, p1_coords, p2_coords, (86, 183, 18), 11)  # Yellow

    trunk_color(img, trunk_angle, shoulder_midpoint_dict, hip_midpoint_dict)


    # Calculate REBA score
    if trunk_angle >= 60:
        critical_limbs.append({"trunk": trunk_angle})
        return 4
    elif trunk_angle <= -20:
        return 3
    elif trunk_angle >= 20:
        return 3
    elif trunk_angle >= 5:
        return 2
    elif trunk_angle <= -5:
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
    

    ###print(f'left leg angle: {left_leg_angle}')
    ###print(f'right leg angle: {right_leg_angle}')

    def leg_color(img, angle, p1, p2):
        if angle >= 60:
            pose_detector.change_line_color(img, 'red', p1, p2)
        elif angle >= 30:
            pose_detector.change_line_color(img, 'yellow', p1, p2)

    leg_color(img, left_leg_angle, left_hip, left_knee)
    leg_color(img, right_leg_angle, right_hip, right_knee)


    if left_leg_angle > right_leg_angle:
        leg_angle = right_leg_angle
    else:
        leg_angle = left_leg_angle

    # Calculate REBA score
    if leg_angle >= 60:
        critical_limbs.append({"leg": leg_angle})
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

    # Color arm based off REBA score
    def upper_arm_color(img, angle, p1, p2):
        if angle >= 90:
            pose_detector.change_line_color(img, 'red', p1, p2)
        elif angle >= 40 or angle <= -40:
            pose_detector.change_line_color(img, 'yellow', p1, p2)

    upper_arm_color(img, left_upper_arm_angle, left_shoulder, left_elbow)
    upper_arm_color(img, right_upper_arm_angle, right_shoulder, right_elbow)

    ###print(f'left upper arm angle: {left_upper_arm_angle}')
    ###print(f'right upper arm angle: {right_upper_arm_angle}')

    if left_upper_arm_angle < right_upper_arm_angle:
        upper_arm_angle = right_upper_arm_angle
    else:
        upper_arm_angle = left_upper_arm_angle

    # Calculate REBA score
    if upper_arm_angle >= 90:
        critical_limbs.append({"upper_arm": upper_arm_angle})
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


    ###print(f'left lower arm angle: {left_lower_arm_angle}')
    ###print(f'right lower arm angle: {right_lower_arm_angle}')

    #Make lines yellow when bad posture
    if left_lower_arm_angle <= 50 or left_lower_arm_angle >= 140:
        pose_detector.change_line_color(img, 'yellow', left_wrist, left_elbow)
    if right_lower_arm_angle <= 50 or right_lower_arm_angle >= 140:
        pose_detector.change_line_color(img, 'yellow', right_wrist, right_elbow)


    if left_lower_arm_angle > right_lower_arm_angle:
        lower_arm_angle = right_lower_arm_angle
    else:
        lower_arm_angle = left_lower_arm_angle

    # Calculate REBA score
    if lower_arm_angle <= 80:
        critical_limbs.append({"lower_arm": lower_arm_angle})
        return 2
    elif lower_arm_angle >= 120:
        critical_limbs.append({"lower_arm": lower_arm_angle})
        return 2
    else: 
        return 1
    

def calc_wrist(direction, index, wrist, elbow, img, pose_detector):
    """
    Finds angle between wrist, shoulder, and elbow to find lower arm angle

    Returns a score based off of the REBA lower arm test 

    NEED TO ADD: wrist twisted
    NEED TESTING, kind of inaccurate because not tracked with knuckle, but index. 
    """

    left_wrist = wrist[0]
    right_wrist = wrist[1]
    left_elbow = elbow[0]
    right_elbow = elbow[1]
    left_index = index[0]
    right_index= index[1]

    # Find angle
    left_wrist_angle = pose_detector.find_angle(img, left_elbow, left_wrist, left_index)
    right_wrist_angle = pose_detector.find_angle(img, right_elbow, right_wrist, right_index)
 
    left_wrist_angle -= 180
    right_wrist_angle -= 180

    ###print(f'left wrist angle: {left_wrist_angle}')
    ###print(f'right wrist angle: {right_wrist_angle}')


    if abs(left_wrist_angle) < abs(right_wrist_angle):
        wrist_angle = left_wrist_angle
    else:
        wrist_angle = right_wrist_angle

    # Calculate REBA score
    if wrist_angle <= -15:
        return 2
    elif wrist_angle >= 15:
        return 2
    else: 
        return 1
    

def second_REBA_score(upper_arm_score, lower_arm_score, wrist_score):
    """
    Calculates the REBA arm score based on upper arm, wrist, and lower arm scores using Table B.
    """
    # Define the REBA Table B as a 2D array based on the image
    reba_table_b = [
        [1, 2, 2, 1, 2, 3],
        [1, 2, 2, 2, 3, 4],
        [3, 4, 5, 4, 5, 5],
        [4, 5, 5, 5, 6, 7],
        [6, 7, 8, 7, 8, 8],
        [7, 8, 8, 8, 9, 9]
    ]

    upper_arm_idx = upper_arm_score - 1
    wrist_idx = (wrist_score - 1) * 3
    lower_arm_idx = lower_arm_score - 1
    
    # Calculate the column index by combining wrist and lower arm indices
    column_idx = wrist_idx + lower_arm_idx

    return reba_table_b[upper_arm_idx][column_idx]


def final_REBA_score(score_a, score_b):
    """
    Calculates the final REBA score using Table C based on Score A and Score B.
    """

    reba_table_c = [
        [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
        [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
        [2, 3, 3, 3, 5, 5, 6, 7, 7, 8, 8, 8],
        [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
        [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
        [6, 6, 6, 6, 7, 8, 9, 9, 10, 10, 10, 10],
        [7, 7, 7, 7, 8, 9, 9, 10, 10, 11, 11, 11],
        [8, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11],
        [9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12],
        [10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12],
        [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    ]

    score_a_idx = score_a - 1
    score_b_idx = score_b - 1

    return reba_table_c[score_a_idx][score_b_idx]



def execute_REBA_test(pose_detector, img):

    # per frame, adds the critical limbs to the list
    global critical_limbs
    critical_limbs = []

    landmark_list = pose_detector.find_position(img)
    direction = pose_detector.find_direction(landmark_list) #based off ear
    neck_result = calc_neck(direction, landmark_list[0], [landmark_list[11], landmark_list[12]], [landmark_list[7], landmark_list[8]], img, pose_detector)
    ###print(f'neck score: {neck_result}')

    trunk_result = calc_trunk(direction, [landmark_list[11], landmark_list[12]], [landmark_list[23], landmark_list[24]], img, pose_detector)
    ###print(f'trunk score: {trunk_result}')

    leg_result = calc_legs(direction, [landmark_list[23], landmark_list[24]], [landmark_list[25], landmark_list[26]], [landmark_list[27], landmark_list[28]], img, pose_detector)
    ###print(f'leg score: {leg_result}')

    # Need to implement step 5; weight of object

    reba_score_1 = first_REBA_score(neck_result, trunk_result, leg_result)

    upper_arm_result = calc_upper_arm(direction, [landmark_list[23], landmark_list[24]], [landmark_list[11], landmark_list[12]], [landmark_list[13], landmark_list[14]], img, pose_detector)
    ###print(f'upper arm score: {upper_arm_result}')

    lower_arm_result = calc_lower_arm(direction, [landmark_list[15], landmark_list[16]], [landmark_list[11], landmark_list[12]], [landmark_list[13], landmark_list[14]], img, pose_detector)
    ###print(f'lower arm score: {lower_arm_result}')

    wrist_result = calc_wrist(direction, [landmark_list[19], landmark_list[20]], [landmark_list[15], landmark_list[16]], [landmark_list[13], landmark_list[14]], img, pose_detector)
    ###print(f'wrist score: {wrist_result}')

    # +1 is temporary because no coupling test, but can assume that most things don't have perfect handle / non existent handle
    reba_score_2 = second_REBA_score(upper_arm_result, lower_arm_result, wrist_result) + 1 

    # Need to create coupling score (handles)
    
    ###print(f'Score 1: {reba_score_1}')
    ###print(f'Score 2: {reba_score_2}')

    final_score = final_REBA_score(reba_score_1, reba_score_2)

    ###print(final_score)

    cv2.putText(img, f'ErgoEye Demo', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 0, 0), 8, cv2.LINE_AA)

    cv2.putText(img, f'REBA Score: {final_score}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (200, 100, 0), 6, cv2.LINE_AA)

    pose_detector.find_critical_poses(img, final_score, 100, critical_limbs)
    
