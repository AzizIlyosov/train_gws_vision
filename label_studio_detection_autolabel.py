# this service is used to run the pretrained model and return the predictions in the format that label studio can understand, 
# so that the predictions can be displayed in the label studio UI
# in order to setup the service,  to run label studio and 
# go <project_name>->settings->Machine Learning->Add Model->"Enter the URL of the model server, validate and save"


from flask import Flask, request, jsonify
import random 
from ultralytics import YOLO
from urllib.parse import urlparse, parse_qs
import cv2 
import numpy as np
import pandas as pd
import os 

detection_model = YOLO("yolo8n960.pt") 

app = Flask(__name__)


 

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
        path = '/'+ query_params.get('d', [None])[0] 
        image  = cv2.imread('/' + path) 
        img1 = image.copy() 
        file_name = path.rsplit('/',1)[1] 
        detections = detection_model.predict(image, verbose=False, agnostic_nms=True)[0]

        bboxes2 = list(detections.boxes.xyxy.cpu().numpy().astype(int))

        names = detections[0].names
        detections[0].plot()

        class_names = list([names[int(cl)] for cl in  detections.boxes.cls.cpu().numpy() ])
    

        result = []
        for cl,[x1, y1, x2, y2] in zip(class_names,  bboxes2):
            
            xmax, ymax = img1.shape[1], img1.shape[0]
            x1,y1=min(x1,x2), min(y1,y2)
            img_height, img_width  = abs(y2-y1), abs(x2-x1)

            r = {
                        'original_width': image.shape[1],
                        'original_height': image.shape[0],
                        'from_name': 'label',
                        'to_name': 'image',
                        'type': 'rectanglelabels',
                        'value': {
                            'rectanglelabels': [ cl],
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
    app.run(debug=True, host='0.0.0.0', port=9093)
