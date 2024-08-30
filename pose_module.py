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
        self.num_poses=4

        # Initialize drawing and pose utilities from MediaPipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
        self.landmark_dict = {0: "nose", 1: "left eye (inner)", 2: "left eye", 3: "left eye (outer)", 4: "right eye (inner)", 5: "right eye", 6: "right eye (outer)", 7: "left ear", 8: "right ear", 9: "mouth (left)", 10: "mouth (right)", 11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow", 15: "left wrist", 16: "right wrist", 17: "left pinky", 18: "right pinky", 19: "left index", 20: "right index", 21: "left thumb", 22: "right thumb", 23: "left hip", 24: "right hip", 25: "left knee", 26: "right knee", 27: "left ankle", 28: "right ankle", 29: "left heel", 30: "right heel", 31: "left foot index", 32: "right foot index"}

    
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
        Calculates coordinates of each joint in the given image frame

        Returns: list of landmarks
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

    def find_direction(self, landmarks):
        """
        Finds direction that person is facing based off landmark confidence scores

        Returns: "left" or "right"

        POSSIBLE EDGE CASE: If joint is covered for some reason, could return wrong direction. 
        Can make it to where if we arent FOR SURE what direction they are facing, we can skip the frame. 
        """
        left_score = 0
        right_score = 0
        
        #add up confidence scores from each side of the body
        for landmark in landmarks:
            if landmark['id'] in [2,7,9,11,13,23,25]:
                left_score += self.results.pose_landmarks.landmark[landmark['id']].visibility
                print(f'{self.landmark_dict[landmark['id']]}: {self.results.pose_landmarks.landmark[landmark['id']].visibility} ')
            if landmark['id'] in [5,8,10,12,14,24,26]:
                right_score += self.results.pose_landmarks.landmark[landmark['id']].visibility
                print(f'{self.landmark_dict[landmark['id']]}: {self.results.pose_landmarks.landmark[landmark['id']].visibility} ')
            

        if left_score > right_score:
            print('facing left')
            return "left"
        else:
            print('facing right')
            return "right"


    def find_angle(self, img, p1, p2, p3, draw=True):
        """
        calculates the angle between three joints in a frame
        """
        ## Get positions of the three joints
        x1, y1 = list(p1.values())[-2:]
        x2, y2 = list(p2.values())[-2:]
        x3, y3 = list(p3.values())[-2:]

        ## Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2,) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360

        """## Makes points green and displays live angle
        if draw:
            cv2.circle(img, (x1,y1), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x3,y3), 5, (0,255,0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
"""
        return angle
    
    def blur_face(self, img):
        """
        Function to blur the detected face
        """

        # Detect faces in the entire image
        mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        results_faces = mp_face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results_faces.detections:
            for detection in results_faces.detections:
                # Get bounding box for the face
                bboxC = detection.location_data.relative_bounding_box
                image_height, image_width, _ = img.shape
                x_min = int(bboxC.xmin * image_width)
                y_min = int(bboxC.ymin * image_height)
                x_max = x_min + int(bboxC.width * image_width)
                y_max = y_min + int(bboxC.height * image_height)
                
                # Extract the region of interest (ROI) and apply Gaussian blur
                roi = img[y_min:y_max, x_min:x_max]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                img[y_min:y_max, x_min:x_max] = blurred_roi

        return img
        
