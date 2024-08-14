import cv2
import mediapipe as mp
import torch

# Initialize MediaPipe and YOLOv5
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to compare distances (You should define your logic here)
def compareDist(dim1, dim2):
    # Example distance metric: Euclidean distance between the centers
    x1, y1, w1, h1 = dim1
    x2, y2, w2, h2 = dim2
    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    return distance

# File list
file_list = ['PoseVideos/lift.png']  # Replace with your image paths

# For each image in the file list
for idx, file in enumerate(file_list):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape

    # Detect objects using YOLOv5
    results = model(image)
    detected_objects = results.xyxy[0]  # Bounding boxes [x_min, y_min, x_max, y_max]

    # Analyze each detected person in the image
    for obj in detected_objects:
        x_min, y_min, x_max, y_max, confidence, class_id = obj
        if class_id == 0:  # Class 0 is 'person' in YOLOv5
            box_width = x_max - x_min
            box_height = y_max - y_min
            detected_boundary = [int(x_min), int(y_min), int(box_width), int(box_height)]

            # Extract the region of interest (ROI) for pose estimation
            roi = image[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Initialize pose estimator
            pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

            # Perform pose estimation on the ROI
            results_pose = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            if results_pose.pose_landmarks:
                # Draw the pose landmarks on the original image, adjusted for the ROI position
                mp_drawing.draw_landmarks(
                    image[int(y_min):int(y_max), int(x_min):int(x_max)],
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

    # Save or display the annotated image
    cv2.imwrite(f'/tmp/annotated_image_{idx}.png', image)
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(4000)

cv2.destroyAllWindows()
