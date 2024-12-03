from flask import Flask, request, jsonify
# write the code to recieve the task from label studio and return the predictions ,immitating the label studio backend ml
import random
# from yolo_track import get_tracking
from ultralytics import YOLO
from urllib.parse import urlparse, parse_qs
import cv2 
import numpy as np
import pandas as pd
import os 

all_images = os.listdir('yolo_as_dataset/images')
all_images_dict = {}

# split by last dot
# all_images = [i.rsplit('.',1)[0] for i in all_images]
for img_name in all_images: 
    with open('yolo_as_dataset/labels/'+img_name.rsplit('.',1)[0]+'.txt', 'r') as f:
        img = cv2.imread('yolo_as_dataset/images/'+img_name)
        # print(img_name)
        lines = f.readlines()
        # split each line by space and convert each value to float 
        lines = [list(map(float,i.split(' '))) for i in lines]
        lines = [[int(i[0]), 
        int((i[1]-i[3]/2) *img.shape[1]),
        int((i[2]-i[4]/2) *img.shape[0]),
        int((i[1]+i[3]/2) *img.shape[1]),
        int((i[2]+i[4]/2) *img.shape[0])
         ] for i in lines]

        all_images_dict[img_name] = {
            'lines':lines,
            'img_shape':img.shape  
            }

# old_labels =  ["7Up 0.25 can","7Up 0.25 nrgb","7Up 0.5 pet","7Up 1 pet","7Up 1.5 pet","Mirinda 0.25 can","Mirinda 0.25 nrgb","Mirinda 0.5 pet","Mirinda 1 pet","Mirinda 1.5 pet","MounDew 0.25 can","MounDew 0.449 can","MounDew 0.5 pet","MounDew 1 pet","Pepsi 0.25 can","Pepsi 0.25 nrgb","Pepsi 0.25 rgb","Pepsi 0.26 pet","Pepsi 0.449 can","Pepsi 0.5 pet","Pepsi 1 pet","Pepsi 1.5 pet","Pepsi 2 pet","Pepsi Z 0.25 can","Pepsi Z 0.5 pet","Pepsi Z 1.5 pet","Tea_black_limon 0.5 pet","Tea_black_limon 1 pet","Tea_black_limon 1.5 pet","Tea_black_peach 0.5 pet","Tea_black_peach 1 pet","Tea_black_peach 1.5 pet","Tea_black_raspberry 0.5 pet","Tea_black_raspberry 1 pet","Tea_black_raspberry 1.5 pet","Tea_black_strawberry 0.5 pet","Tea_black_strawberry 1 pet","Tea_black_strawberry 1.5 pet","Tea_green_limon 0.5 pet","Tea_green_limon 1 pet","Tea_green_limon 1.5 pet","Tea_green_peach 0.5 pet","Tea_green_peach 1 pet","Tea_green_peach 1.5 pet","Tea_green_raspberry 0.5 pet","Tea_green_raspberry 1 pet","Tea_green_raspberry 1.5 pet","Tea_green_strawberry 0.5 pet","Tea_green_strawberry 1 pet","Tea_green_strawberry 1.5 pet"]
old_labels =  ["7Up 0.25 can","7Up 0.25 nrgb","7Up 0.5 pet","7Up 1 pet","7Up 1.5 pet","Mirinda 0.25 can","Mirinda 0.25 nrgb","Mirinda 0.5 pet","Mirinda 1 pet","Mirinda 1.5 pet","MounDew 0.25 can","MounDew 0.449 can","MounDew 0.5 pet","MounDew 1 pet","Pepsi 0.25 can","Pepsi 0.25 nrgb","Pepsi 0.25 rgb","Pepsi 0.26 pet","Pepsi 0.449 can","Pepsi 0.5 pet","Pepsi 1 pet","Pepsi 1.5 pet","Pepsi 2 pet","Pepsi Z 0.25 can","Pepsi Z 0.5 pet","Pepsi Z 1.5 pet","Tea_black_limon 0.5 pet","Tea_black_limon 1 pet","Tea_black_limon 1.5 pet","Tea_black_peach 0.5 pet","Tea_black_peach 1 pet","Tea_black_peach 1.5 pet","Tea_black_raspberry 0.5 pet","Tea_black_raspberry 1 pet","Tea_black_raspberry 1.5 pet","Tea_black_strawberry 0.5 pet","Tea_black_strawberry 1 pet","Tea_black_strawberry 1.5 pet","Tea_green_limon 0.5 pet","Tea_green_limon 1 pet","Tea_green_limon 1.5 pet","Tea_green_peach 0.5 pet","Tea_green_peach 1 pet","Tea_green_peach 1.5 pet","Tea_green_raspberry 0.5 pet","Tea_green_raspberry 1 pet","Tea_green_raspberry 1.5 pet","Tea_green_strawberry 0.5 pet","Tea_green_strawberry 1 pet","Tea_green_strawberry 1.5 pet"]
detection_model = YOLO("detection_best.pt")
classifications_model = YOLO("classification_best.pt")

app = Flask(__name__)


def iou(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        x5, y5 = max(x1, x3), max(y1, y3)
        x6, y6 = min(x2, x4), min(y2, y4)
        if x5>=x6 or y5>=y6:
            return 0
        intersection = (x6-x5)*(y6-y5)
        union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - intersection
        return intersection/union


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'UP'})

@app.route('/setup', methods=['POST'])
def setup():
    return jsonify({'status': 'UP'})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get the task data from the request
    tasks = request.json  
  
    for  task in tasks['tasks']: 
        parsed_url = urlparse(task['data']['image']) 
        query_params = parse_qs(parsed_url.query)
        path = query_params.get('d', [None])[0] 
        image  = cv2.imread('/' + path)
        print('path ', path)
        img1 = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        img2= image.copy()
        file_name = path.rsplit('/',1)[1] 
        detections = detection_model(img2, verbose=False)
        
        # cut_images = []
        bboxes= []
        
        for detection in detections[0].boxes: 
            x1, y1, x2, y2 = detection.xyxy.cpu().numpy()[0].astype(int) 
            bboxes.append([x1, y1, x2, y2])

        bboxes2 = []
        removes = []

        class_names =[]
        if  file_name in all_images_dict:
            det1 = all_images_dict[file_name]
            class_names = [old_labels[i[0]] for i in  det1['lines']]
            bboxes2 = [i[1:] for i in  det1['lines']] 
            
            for i,bbox in enumerate(bboxes):
                for bbox2 in bboxes2:
                    # find iou of bbox and bbox2
                    coef = iou(bbox, bbox2)
                    if coef>0.1: 
                        removes.append(i)
                        break
        
        bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in removes] 

        for bbox in bboxes:
            bboxes2.append(bbox)
            class_names.append('no_label')
        

        result = []
        for cl,[x1, y1, x2, y2] in zip(class_names,  bboxes2):
            
            xmax, ymax = img2.shape[1], img2.shape[0]
            x1,y1=min(x1,x2), min(y1,y2)
            img_height, img_width  = abs(y2-y1), abs(x2-x1)

            r = {
                        'oroginal_width': image.shape[1],
                        'oroginal_height': image.shape[0],
                        'from_name': 'label',
                        'to_name': 'image',
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': [ cl ],
                            'x':  float(x1)/image.shape[1] *100 ,
                            'y':  float(y1 )/image.shape[0]*100 ,
                            'width': float(img_width )/image.shape[1]*100,
                            'height': float(img_height)/image.shape[0]*100,
                        },
                    }
            result.append(r)

        predictions = [{ 'result':result,   } ] 

     
    response = {
        'results': predictions,
        'model_version': '10.02.01'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9092)
