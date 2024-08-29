import cv2
import mediapipe as mp
from pose_module import poseDetector
from REBA_calc import calcREBAPose

def singleperson():
    # Creates video object
    # cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pose_detector = poseDetector()

    ## Processes image frames
    ## while True:
    img = cv2.imread('PoseVideos/14.png')


    """success, img = cap.read()

    # Break the loop if the video ends
    if not success:
        print("Finished processing video.")
        break

    if img is None:
        print("Warning: Captured frame is None.")
        continue"""
    

    
    img = pose_detector.find_pose(img)
    landmark_list = pose_detector.find_position(img)
    
    reba = calcREBAPose(img, landmark_list)
    #reba.calc_upper_arm(pose_detector.find_angle(img, 13, 11, 23))

    cv2.imshow("Image", img)
    cv2.waitKey(5000)


if __name__ == "__main__":
    singleperson()
