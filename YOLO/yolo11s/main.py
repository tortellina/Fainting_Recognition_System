from ultralytics import YOLO
import cv2



MODEL_PATH = r"C:\Users\vitto\Desktop\EXAMS\3 SEMESTER\2 -project COMPUTER VISION\my_model\train\weights\best.pt" #model path
VIDEO_PATH = r"C:\Users\vitto\Downloads\20250709_210023 (1).mp4" #input video
OUTPUT_PATH = r"yolooo.mp4" #output video path

#upload model
model = YOLO(MODEL_PATH)

#open video
cap = cv2.VideoCapture(VIDEO_PATH)
print(f'your video is being analyzed, please wait')

if not cap.isOpened():
    print(f" ERROR: impossible to open video '{VIDEO_PATH}'")
    exit()

#video parameters
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


#output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print(f" ERROR: impossible to create output video '{OUTPUT_PATH}'")
    cap.release()
    exit()

#video analysis
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    
    
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    #show video in real time analysis
    cv2.imshow("YOLOv8", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print(f'your video is ready')
