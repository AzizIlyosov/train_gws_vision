from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

ObjectDetectorWriter = object_detector.MetadataWriter

  

model_type="FLOAT32"
model_name = "best_float32"
# _MODEL_PATH = "yolo/runs/detect/train22/weights/best_saved_model/best_float32.tflite"
_MODEL_PATH = "runs/detect/PEPSI_yolov8n_960_14926/weights/best_saved_model/"+model_name+ ".tflite"
# Task Library expects label files that are in the same format as the one below.
# _LABEL_FILE = "best_float32_labels.txt"

# _SAVE_TO_PATH = "yolo_with_metadata.tflite"
_SAVE_TO_PATH = model_name+"_meta.tflite" 

if model_type=="FLOAT32":
    _INPUT_NORM_MEAN =127.5
    _INPUT_NORM_STD = 127.5
else:
    _INPUT_NORM_MEAN = 0 # 127.5
    _INPUT_NORM_STD = 255# 127.5


# Create the metadata writer.
writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
    [])

 
 
print(writer.get_metadata_json())
# print(writer.__dir__() )
 
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)