from ultralytics import YOLO
import os 
import os

base_path=os.path.dirname(__file__)
base_path_model = base_path+ '/runs/detect'

folders_numbers =['2759', ]
folders = [folder for folder in  os.listdir(base_path_model) if (not(folder[:3]=='val' or folder[:3]=='pre') and folder.split('_')[-1] in folders_numbers)]
# Load a model
 
print('base_path ', base_path)


res = {}

for folder in folders:
    try:
        model = YOLO(base_path_model+'/'+folder+'/weights/best.pt')  # load an official model
        metrics = model.val(data='data.yaml')  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        print(metrics.box.map)
        res[folder]= metrics.box.map
    except Exception as ex:
        print('not able to load the model '+ base_path_model+'/'+folder+'/weights/best.pt')
        print(ex)
        exit(0)
        pass 

print(res)

    