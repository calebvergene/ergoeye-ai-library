import cv2
import mediapipe as mp
import math


class poseDetector():
    """
    creates instance of video
    """
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

    
    def find_pose(self, img, draw=True):
        """
        returns a frame of the pose with joint points and lines drawn
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.pose.process(imgRGB)
        ## Draws data from joint points on body in live video
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        """
        calculates coordinates of each joint in the given image frame
        """
        self.landmark_list = []

        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                # convert to pixel value
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append({"id":id, "x":cx, "y":cy})
                cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return self.landmark_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """
        calculates the angle between three joints in a frame
        """
        ## Get positions of the three joints
        x1, y1 = list(self.landmark_list[p1].values())[-2:]
        x2, y2 = list(self.landmark_list[p2].values())[-2:]
        x3, y3 = list(self.landmark_list[p3].values())[-2:]

        ## Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2,) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360

        ## Makes points green and displays live angle
        if draw:
            cv2.circle(img, (x1,y1), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x3,y3), 5, (0,255,0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        return angle
