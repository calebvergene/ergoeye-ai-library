import cv2
import mediapipe as mp
from pose_module import poseDetector
from REBA_calc import execute_REBA_test

def singleperson():
    """
    Function for executing ergonomic assesments with multiple people in it
    """
    # Creates video object
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pose_detector = poseDetector()

    ## Processes image frames
    while True:
        #raw_img = cv2.imread('PoseVideos/13.png')


        success, img = cap.read()

        # Break the loop if the video ends
        if not success:
            print("Finished processing video.")
            break

        if img is None:
            print("Warning: Captured frame is None.")
            continue
    
        img = pose_detector.find_pose(img)
        try:
            execute_REBA_test(pose_detector, img)
        except:
            continue

        cv2.imshow("Image", img)
        #cv2.waitKey(5000)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    singleperson()
