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
model = YOLO("yolo11n.pt")

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

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            if 'person' in c:
                result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
                if result>=0:
                   cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                   cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)
                   if people_count.count(track_id)==0:
                      people_count.append(track_id)
    p_counter=len(people_count)                  
    cvzone.putTextRect(frame,f'People_Counter:-{p_counter}',(50,60),2,2)                  
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
