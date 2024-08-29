import cv2
import torch
from pose_module import poseDetector
from REBA_calc import calcREBAPose

def multiperson():
    # Initialize YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Initialize the poseDetector
    pose_detector = poseDetector()

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
                # pose_detector.blur_face(roi_with_pose)
                landmark_list = pose_detector.find_position(roi_with_pose)
                reba = calcREBAPose(roi_with_pose, landmark_list)
                
                #reba.calc_upper_arm(pose_detector.find_angle(roi_with_pose, 13, 11, 23))


                
                
    return image

image = multiperson()

# Save or display the annotated image
# cv2.imwrite(f'/tmp/annotated_image_{idx}.png', image)
cv2.imshow('Annotated Image', image)
cv2.waitKey(4000)

cv2.destroyAllWindows()