import cv2
import mediapipe as mp


mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/1.mp4')

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(40)