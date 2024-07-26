import cv2
import mediapipe as mp
mpDraw = mp.solutions.drawing_utils


## These are the new params for Pose()
"""def __init__(self,static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):"""
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Creates video object
cap = cv2.VideoCapture('PoseVideos/1.mp4')


## Processes image frames
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    ## Draws data from joint points on body in live video
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(40)
