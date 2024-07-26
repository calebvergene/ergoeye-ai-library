import cv2
import mediapipe as mp


class poseDetector():
    def __init__(self,static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.static_image_mode=False
        self.model_complexity=1
        self.smooth_landmarks=True
        self.enable_segmentation=False
        self.smooth_segmentation=True
        self.min_detection_confidence=0.5
        self.min_tracking_confidence=0.5

        # Initialize drawing and pose utilities from MediaPipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)

    
    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.pose.process(imgRGB)
        ## Draws data from joint points on body in live video
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

        """for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            # convert to pixel value
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)"""

    


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
        
        img = detector.findPose(img)
    
        cv2.imshow("Image", img)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
