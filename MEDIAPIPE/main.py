import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import vision
import time


MODEL_PATH = "C:/Users/vitto/Desktop/pose_landmarker_full.task" #path to the model
VIDEO_PATH = r"C:\Users\vitto\Downloads\20250709_210023 (1).mp4"  #input video
output_path = "MediaPipe_fainting.mp4"  #output video name

#options for the model
options = PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    output_segmentation_masks=True
)

#open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

#for output the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


#analyse video
with PoseLandmarker.create_from_options(options) as landmarker:
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #conversion of the video in mediapipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        #millisecond timestamp needed for mediapipe
        timestamp = int((frame_index / fps) * 1000)

        #analyse the frame
        result = landmarker.detect_for_video(mp_image, timestamp)

        #show segmentation mask
        if result.segmentation_masks:
            mask = result.segmentation_masks[0].numpy_view()
            mask_img = (mask * 255).astype("uint8")
            mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            
        frame_index += 1

        #none detected
        if len(result.pose_landmarks) == 0:
            cv2.putText(frame, "None detected", (60, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.imshow("Segmentation Mask Overlay", frame)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
       
       #if a person is showing 
        landmarks = result.pose_landmarks[0]

        l_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        dy = abs(shoulder_y - hip_y)

        #bounding box
        h, w = frame.shape[:2]
        points_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        x_values = [p[0] for p in points_px]
        y_values = [p[1] for p in points_px]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"PERSON", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #person classification
        if dy < 0.1:
            status = "fainted"
            color = (0, 0, 255)
        else:
            status = "standing"
            color = (0, 255, 0)

        #show status in the display
        cv2.putText(frame, status, (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)

        #show video
        overlay = cv2.addWeighted(frame, 0.7, mask_bgr, 0.3, 0)
        cv2.imshow("Segmentation Mask Overlay", overlay)
        out.write(overlay)

      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    out.release()

    cv2.destroyAllWindows()