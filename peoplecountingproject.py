from ultralytics import YOLO
import cv2
import cvzone
import datetime

model = YOLO('yolo11n.pt')
mask = cv2.VideoCapture('masked.mp4')
cap = cv2.VideoCapture('people.mp4')
thingstodetect = ["person"]

yy = 400
line_start = (0, yy)
line_end = (1280, yy)

offset = 20        
detected = []   
people_count = 0

while True:
    ret , frames = cap.read()
    ret2 , frames2 = mask.read()


    if not ret:
        break

    cv2.line(frames, line_start, line_end, (0, 255, 0), 2)


    magic = model.track(frames2, persist=True,tracker="bytetrack.yaml")
    for r in magic:
        boxes = r.boxes
        if r.boxes.id is not None:
            for box , track_id in zip(boxes,r.boxes.id):
                cls = int(box.cls[0])
                things = model.names[cls]

                if things not in thingstodetect:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0]  
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                centerx = x1 + w // 2
                centery = y1 + h // 2

                cvzone.cornerRect(frames ,(x1, y1, w, h))

                conf = int(box.conf[0] * 100)
                label = f"{model.names[cls]} {conf}%"

                cvzone.putTextRect(frames, label, (x1, y1 - 10), scale=1, thickness=1)


                if (yy - offset < centery < yy + offset) and (track_id not in detected):
                    people_count += 1
                    detected.append(track_id)

   

    now = datetime.datetime.now()
    cv2.putText(frames, f'Time>{now.hour}:{now.minute}:{now.second}', (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frames, f'PeoplePassed:{people_count}', (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("detecyting",frames)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
