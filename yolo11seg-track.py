import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("yolo11n-seg.pt")
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('m.avi')
count=0
area=[(222,118),(194,337),(799,300),(728,112)]

cy1=178
offset=10
enter_student=[]
people_count=[]
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xyxy.int().cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    class_ids = results[0].boxes.cls.int().cpu().tolist()
    masks = results[0].masks
    if masks is not None:
        
        clss = results[0].boxes.cls.cpu().tolist()
        masks = masks.xy
        overlay = frame.copy()

        for box,track_id,class_id,mask in zip(boxes, track_ids,class_ids,masks):
            # Convert mask points to integer
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            if 'person' in c:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(overlay, [mask], color=(0,0, 255))
                # Draw the mask on the frame (filled polygon)
#               cv2.polylines(overlay, [mask], isClosed=True, color=(0, 255, 0), thickness=2)  # Green outline

                # Optionally, draw the bounding box too
                cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)
                # Optionally, draw the class label
        alpha = 0.5  # Transparency factor (0 = invisible, 1 = fully visible)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

   
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

