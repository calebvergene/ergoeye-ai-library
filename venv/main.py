import cv2
import mediapipe as mp
from pose_module import poseDetector

def main():
    # Creates video object
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    detector = poseDetector()

    ## Processes image frames
    while True:
        success, img = cap.read()

        # Break the loop if the video ends
        if not success:
            print("Finished processing video.")
            break

        if img is None:
            print("Warning: Captured frame is None.")
            continue
        
        img = detector.find_pose(img)
        landmark_list = detector.find_position(img)
        
    
        cv2.imshow("Image", img)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
