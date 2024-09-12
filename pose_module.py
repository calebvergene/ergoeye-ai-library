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
        self.min_detection_confidence=0.7
        self.min_tracking_confidence=0.7
        self.num_poses=4


        # Initialize drawing and pose utilities from MediaPipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
        self.landmark_dict = {0: "nose", 1: "left eye (inner)", 2: "left eye", 3: "left eye (outer)", 4: "right eye (inner)", 5: "right eye", 6: "right eye (outer)", 7: "left ear", 8: "right ear", 9: "mouth (left)", 10: "mouth (right)", 11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow", 15: "left wrist", 16: "right wrist", 17: "left pinky", 18: "right pinky", 19: "left index", 20: "right index", 21: "left thumb", 22: "right thumb", 23: "left hip", 24: "right hip", 25: "left knee", 26: "right knee", 27: "left ankle", 28: "right ankle", 29: "left heel", 30: "right heel", 31: "left foot index", 32: "right foot index"}

    
    def find_pose(self, img, draw=True):
        """
        returns a frame of the pose with joint points and lines drawn. 

        A LOT of this code is just code to draw the lines inbetween the head.
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.pose.process(imgRGB)

        # Draws data from joint points on body in live video, excluding face landmarks (0 to 10)
        if self.results.pose_landmarks:
            if draw:
                line_spec = self.mpDraw.DrawingSpec(color=(86, 183, 18), thickness=11)  # Green lines
                landmark_spec = self.mpDraw.DrawingSpec(color=(1, 183, 86), thickness=11)  # Red points

                # Get the landmark list
                pose_landmarks = self.results.pose_landmarks.landmark

                # Calculate the midpoints between shoulders and ears
                left_shoulder = pose_landmarks[11]  # Left shoulder
                right_shoulder = pose_landmarks[12]  # Right shoulder
                left_ear = pose_landmarks[7]  # Left ear
                right_ear = pose_landmarks[8]  # Right ear

                # Midpoint between shoulders
                mid_shoulders = (
                    (left_shoulder.x + right_shoulder.x) / 2,
                    (left_shoulder.y + right_shoulder.y) / 2
                )

                # Midpoint between ears
                mid_ears = (
                    (left_ear.x + right_ear.x) / 2,
                    (left_ear.y + right_ear.y) / 2
                )

                # Convert these midpoints to image coordinates
                h, w, _ = img.shape
                mid_shoulders_coord = (int(mid_shoulders[0] * w), int(mid_shoulders[1] * h))
                mid_ears_coord = (int(mid_ears[0] * w), int(mid_ears[1] * h))
                left_ear_coord = (int(left_ear.x * w), int(left_ear.y * h))
                right_ear_coord = (int(right_ear.x * w), int(right_ear.y * h))

                # Draw connections from the middle of the shoulders to the middle of the ears
                cv2.line(img, mid_shoulders_coord, mid_ears_coord,  (86, 183, 18), 11)
                # Draw lines from mid-ears to left and right ears
                cv2.line(img, mid_ears_coord, left_ear_coord, (86, 183, 18), 11)
                cv2.line(img, mid_ears_coord, right_ear_coord, (86, 183, 18), 11)

                # Now draw the usual body pose connections, excluding face connections (0-10)
                connections = [
                conn for conn in self.mpPose.POSE_CONNECTIONS
                if conn[0] > 10 and conn[1] > 10  # Exclude face connections
            ]
                self.mpDraw.draw_landmarks(
                    img, 
                    self.results.pose_landmarks,
                    connections,  # Use filtered connections
                    connection_drawing_spec=line_spec,
                    landmark_drawing_spec=None
                )
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
        
        # Adds up confidence scores from each side of the body
        for landmark in landmarks:
            if landmark['id'] in [2,7,9,11,13,23,25]:
                left_score += self.results.pose_landmarks.landmark[landmark['id']].visibility
                ### print(f'{self.landmark_dict[landmark['id']]}: {self.results.pose_landmarks.landmark[landmark['id']].visibility} ')
            if landmark['id'] in [5,8,10,12,14,24,26]:
                right_score += self.results.pose_landmarks.landmark[landmark['id']].visibility
                ### print(f'{self.landmark_dict[landmark['id']]}: {self.results.pose_landmarks.landmark[landmark['id']].visibility} ')
            

        if left_score > right_score:
            return "left"
        else:
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
    

    def change_line_color(self, img, color, p1, p2):
        """
        Changes line color between two landmarks on the body
        """
        
        if (self.results.pose_landmarks.landmark[p1['id']].visibility > 0.7 and 
        self.results.pose_landmarks.landmark[p2['id']].visibility > 0.7):

            # Ensure the coordinates are integers
            p1_coords = (int(p1['x']), int(p1['y']))
            p2_coords = (int(p2['x']), int(p2['y']))

            # Draw the line with the specified color
            if color == "yellow":
                cv2.line(img, p1_coords, p2_coords, (42, 212, 227), 11)  # Yellow
            elif color == "red":
                cv2.line(img, p1_coords, p2_coords, (61, 61, 255), 11)  # Red