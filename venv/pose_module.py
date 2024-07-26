import cv2
import mediapipe as mp

# Initialize drawing and pose utilities from MediaPipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Creates video object
cap = cv2.VideoCapture('PoseVideos/1.mp4')


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


    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = pose.process(imgRGB)

    ## Draws data from joint points on body in live video
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            # convert to pixel value
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(20)