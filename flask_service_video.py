from flask import Flask, request, jsonify
# write the code to recieve the task from label studio and return the predictions ,immitating the label studio backend ml
import random
from yolo_track import get_tracking
from urllib.parse import urlparse, parse_qs

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
        
        # print('task ', task)
 
        parsed_url = urlparse(task['data']['video'])
        query_params = parse_qs(parsed_url.query)
        path = query_params.get('d', [None])[0]
        outputs  = get_tracking(path, 5)
        
        result = []
        for obj_index in outputs[0]:
            res = {}
            object_frames = outputs[0][obj_index]
            sequence = []
            value= {}

            
            for object_frame in object_frames:
                x,y,w,h,frame_order = object_frame
                d= {
                    'enabled':True,
                    'rotation':0,
                    'frame':frame_order,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h  
                }
                sequence.append(d)
            
            d= {
                    'enabled':False,
                    'rotation':0,
                    'frame':frame_order+1,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h  
                }
            sequence.append(d)
            
            value['sequence']= sequence
            value['labels'] =['Pepsi 1.5 pet']
            # value["framesCount"]= len(sequence)
            res={
                
                'id':str(obj_index),
                "value":value,
                "from_name": "box",
                "to_name": "video",
                "type": "videorectangle",
                "score": 0.95,
                 }
            
                
            result.append(res)

        predictions = [{ 'result':result, 'frame_count':outputs[1] } ] 

     
    response = {
        'results': predictions,
        'model_version': '10.02.01'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9091)
