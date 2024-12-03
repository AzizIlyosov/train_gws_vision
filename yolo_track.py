from collections import defaultdict

import cv2
import numpy as np
import datetime 
import os 
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
 

 
def get_tracking(video_path, steps_per_frame=3):
    model = YOLO("best.pt")

    video_path = '/' + video_path 

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    track_history = defaultdict(lambda: []) 
    frame_order = 0  
    fr_shape = []
    while cap.isOpened():
        
        success, frame = cap.read() 
        
        if success:
            frame_order+=1

            if  frame_order%steps_per_frame!=0:
                continue
                        
            video_height, video_width, video_channels  = frame.shape
            results = model.track(frame, persist=True , )
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = 100* box[0]/video_width,100* box[1]/video_height,100* box[2]/video_width, 100*box[3]/video_height
                x,y = x-w/2,y-h/2 
                track = track_history[track_id]
                track.append(  (float(x), float(y) , float(w),  float(h), frame_order) ) 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    return track_history, frame_count  






def custom_plot(frame2, bboxes):
    video_height, video_width, video_channels  = frame2.shape
    frame = frame2.copy() 
    count=0 
    for bbox in bboxes.cpu() :
        x,y,w,h = [int(i) for i in bbox]
        x=x-w//2
        y=y-h//2
        # x, y, w, h = 100* box[0]/video_width,100* box[1]/video_height,100* box[2]/video_width, 100*box[3]/video_height
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h),  (255, 0, 0),   5 )
        count+=1 
    cv2.imshow('mywindow', frame)
 



if __name__=='__main__':

    

    model = YOLO("best.pt")
    video_path = "v.mp4"
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])




    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True   )
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # print('this is track ids ', len(track_ids),  )
            # print('this is boxes  ids ', boxes.shape)
            # continue
            annotated_frame = results[0].plot()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            # cv2.imshow("YOLOv8 Tracking", annotated_frame)
            print(boxes[0,: ])
            custom_plot(frame, boxes)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # print('this is track_history', track_history)

    cap.release()
    cv2.destroyAllWindows()