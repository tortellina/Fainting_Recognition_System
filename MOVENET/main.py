import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


#video path
video_path = r"C:\Users\vitto\Downloads\20250709_210023 (1).mp4" #CHANGE WITH YOUR VIDEOPATH
output_path = "movenet_fainting.mp4"  #CHANGE output video name



#upload model movenet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']


def detect_pose(image):
    input_image = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)

    results = movenet(input_image)
    keypoints = results['output_0'].numpy()[0][0]
    
    return keypoints


def is_person_standing(keypoints):
    shoulder_y = (keypoints[5][0] + keypoints[6][0]) / 2
    hip_y = (keypoints[11][0] + keypoints[12][0]) / 2
    return abs(hip_y - shoulder_y)

 
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

#for output the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = detect_pose(img)
    print(keypoints)

    #bounding box
    h, w = frame.shape[:2]
    try:
        valid_points = [
            (int(x * w), int(y * h))
            for y, x, conf in keypoints
            
        ]
        x_coords = [pt[0] for pt in valid_points]
        y_coords = [pt[1] for pt in valid_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Draw the bounding box and label
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "PERSON", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    except:
        pass


    if is_person_standing(keypoints) == 0:
        label = 'none detected'
    elif is_person_standing(keypoints) > 0.1:
        label = "standing"
    else:
        label = "fainted"

    cv2.putText(frame, label, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    # Mostra il frame

    

    cv2.imshow('Pose Detection', frame)
    out.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()