from ultralytics import YOLO
import tensorflow as tf 
from PIL import Image
import numpy  as np
import cv2 
# Load your TensorFlow Lite model, perform the inference, and get the output tensor
interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

# Get the input and output shapes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the image and run inference
image_data = Image.open("img.png")

[ width, height, ] = image_data.size
length = max ((height, width)) 
scale = length / int(960) 
image_data = image_data.resize((960, 960))
image_data = np.array(image_data).astype(np.float32)
image_data = np.expand_dims(image_data, axis=0)
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()

# Get the raw output tensor
print('this is input details ', input_details  )
print('this is input details ', output_details )
print('\n\n\n\n\n')

output_data = interpreter.get_tensor(output_details[0]['index'])
print('this is  output data ', output_data)
exit()


outputs = np.array([np.transpose(output_data[0].astype(np.float32))])
print('this is output data ', output_data.shape)

conf = 0.3
iou=0.5
multi_class=True 

rows = outputs.shape[1]
boxes = []
scores = []
class_ids = []
for i in range(rows):
    classes_scores = outputs[0][i][4:]
    (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
    if maxScore >= 0.25:
        box = [
            int((outputs[0][i][0] - (0.5 * outputs[0][i][2])) * scale), 
            int((outputs[0][i][1] - (0.5 * outputs[0][i][3])) * scale),
            int(outputs[0][i][2] * scale), 
            int(outputs[0][i][3] * scale) 
            ]
        boxes.append(box)
        scores.append(maxScore)
        class_ids.append(maxClassIndex)
result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf, iou, 0.5)  
bboxes =[boxes[index] for index in result_boxes]
# if  multi_class:
#     return class_ids, bboxes
# return bboxes
print('this  is bboxes  ', bboxes)

exit()

output_data= np.transpose(output_data, (0,2,1))
print('shape of output data ', output_data.shape)
box = tf.expand_dims(output_data[..., :4], axis=2)
print('shape of the boxes ', box.shape)
scores =    output_data[..., 4:] 
print('this is shapes ', scores.shape)
# exit()

# Apply non-maximum suppression (NMS)
detections = tf.image.combined_non_max_suppression(
    boxes=box,
    scores= scores,
    max_output_size_per_class=100,
    max_total_size=100,
    iou_threshold=0.45,
    score_threshold=0.25
)

# Filter out detections with low confidence scores
boxes = detections.nmsed_boxes[detections.nmsed_scores[:, 0] > 0.25]
scores = detections.nmsed_scores[:, 0][detections.nmsed_scores[:, 0] > 0.25]
classes = detections.nmsed_classes[detections.nmsed_scores[:, 0] > 0.25]

# Rescale the bounding box coordinates to the size of the original image
boxes = boxes * 960

# Print the final detections
print(boxes)
print(scores)
print (classes)