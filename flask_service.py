from flask import Flask, request, jsonify
# write the code to recieve the task from label studio and return the predictions ,immitating the label studio backend ml
import random
# from yolo_track import get_tracking
from ultralytics import YOLO
from urllib.parse import urlparse, parse_qs
import cv2 
import numpy as np
import pandas as pd
old_labels = ['Pepsi 2 pet','Pepsi 1.5 pet','Pepsi 1 pet','Pepsi 0.5 pet','Pepsi 0.26 pet','Pepsi 0.449 can','Pepsi 0.25 can','Pepsi 0.25 rgb','Pepsi 0.25 nrgb','Pepsi Z 1.5 pet','Pepsi Z 1 pet','Pepsi Z 0.25 can','Pepsi Z 0.25 nrgb','Pepsi Z 0.5 pet','Mirinda 1.5 pet','Mirinda 1 pet','Mirinda 0.5 pet','Mirinda 0.25 can','Mirinda 0.25 nrgb','7Up 1.5 pet','7Up 1 pet','7Up 0.5 pet','7Up 0.25 can','7Up 0.25 nrgb','MounDew 1 pet','MounDew 0.5 pet','MounDew 0.25 can','Tea_green_strawberry 1 pet','Tea_green_strawberry 0.5 pet','Tea_green_limon 1.5 pet','Tea_green_limon 1 pet','Tea_green_limon 0.5 pet','Tea_green_peach 1 pet','Tea_green_peach 0.5 pet','Tea_black_limon 1 pet','Tea_black_limon 0.5 pet','Tea_black_raspberry 1.5 pet','Tea_black_raspberry 1 pet','Tea_green_raspberry 0.5 pet','Tea_black_peach 1 pet','Tea_black_peach 0.5 pet','RockStar Original 0.449 can','RockStar Original 0.25 can','RockStar Guana 0.449 can','RockStar Persik 0.449 can','Adrenalin 0.449 can','Adrenalin 0.25 can','Adrenalin Classic 0.449 can','Adrenalin Mango 0.449 can','Adrenalin Classic 0.25 can']

detection_model = YOLO("detection_best.pt")
classifications_model = YOLO("classification_best.pt")

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

    # get path of the model 
    # print('this is tasks ', tasks )


    for  task in tasks['tasks']:
        
        # print('task ', task)
 
        parsed_url = urlparse(task['data']['image'])
        # print('parsed_url ', parsed_url)
        query_params = parse_qs(parsed_url.query)
        path = query_params.get('d', [None])[0]
        # print('path ', path) 
        image  = cv2.imread('/'+ path)
        img1 = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print('image ', image)
        img2= image.copy()
        detections = detection_model(img2, verbose=False)
        
        # print('detections ', detections[0].boxes)
         
        # cut images by detections 
        cut_images = []
        bboxes= []
        # cut each object from image by result of ultralytics and sent to classification model
        for detection in detections[0].boxes: 
            x1, y1, x2, y2 = detection.xyxy.cpu().numpy()[0].astype(int) 
            bboxes.append([x1, y1, x2, y2])
            cut_image = img1[y1:y2, x1:x2]
            cut_images.append(cut_image) 
        
        results = classifications_model.predict(cut_images, verbose=False)
        probs = results[0].probs.cpu().numpy()
        
        class_names = []
        for result in results:
            class_name = result.names[int(result.probs.argmax().cpu())]
            class_names.append(class_name)
        
        result = []
        for cl,bbox in zip(class_names, bboxes):
            if not(cl in old_labels):
                cl = 'no_label'
            x1, y1, x2, y2 = bbox
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
    app.run(debug=True, host='0.0.0.0', port=9091)
