"""
- need to calculate all REBA steps per frame

- create a class that takes in (image,landmark_list) as input, then runs all calculations for REBA steps with each body test
- then, calculate the final score and display the score for each frame
"""

def calc_neck():
    return


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

    """neck_direction = 'x'
    neck_angle = pose_detector.find_angle(img)"""