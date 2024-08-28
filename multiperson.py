import cv2
import mediapipe as mp
import torch
from pose_module import poseDetector

def multiperson():
    # Initialize YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    # Initialize the poseDetector
    pose_detector = poseDetector()

    def blur_face(image):
        """
        Function to blur the detected face
        """

        # Detect faces in the entire image
        results_faces = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results_faces.detections:
            for detection in results_faces.detections:
                # Get bounding box for the face
                bboxC = detection.location_data.relative_bounding_box
                image_height, image_width, _ = image.shape
                x_min = int(bboxC.xmin * image_width)
                y_min = int(bboxC.ymin * image_height)
                x_max = x_min + int(bboxC.width * image_width)
                y_max = y_min + int(bboxC.height * image_height)
                
                # Extract the region of interest (ROI) and apply Gaussian blur
                roi = image[y_min:y_max, x_min:x_max]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                image[y_min:y_max, x_min:x_max] = blurred_roi

        return image

    # File list
    file_list = ['PoseVideos/lift.png']  # Replace with your image paths

    # Process each image in the file list
    for file in file_list:
        image = cv2.imread(file)

        # Detect objects using YOLOv5
        results = model(image)
        detected_objects = results.xyxy[0]  # Bounding boxes [x_min, y_min, x_max, y_max]

        # Analyze each detected person in the image
        for obj in detected_objects:
            x_min, y_min, x_max, y_max, confidence, class_id = obj
            if class_id == 0:  # Class 0 is 'person' in YOLOv5
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Extract the region of interest (ROI) for pose estimation
                roi = image[int(y_min):int(y_max), int(x_min):int(x_max)]

                # Perform pose estimation on the ROI
                roi_with_pose = pose_detector.find_pose(roi)

                blur_face(roi_with_pose)
                
    return image

image = multiperson()

# Save or display the annotated image
# cv2.imwrite(f'/tmp/annotated_image_{idx}.png', image)
cv2.imshow('Annotated Image', image)
cv2.waitKey(4000)

cv2.destroyAllWindows()