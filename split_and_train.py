import pandas as pd 
import os 
import random 
import shutil
from ultralytics import YOLO 
import zipfile
import os 
import pandas as pd
import yaml
import os 
import pandas as pd
import yaml
 



# extract from zip files

def make_yolo_data(base_path = 'all_projects'):
    

    # all_data = [f for f in os.listdir(base_path)  if  'project-2' in f  ]
    all_data = os.listdir(base_path)
    all_data = [i for i in all_data  ]#if not( i=="project_2_6.zip" or i=="project_2_5.zip") ] 
    for f in os.listdir('yolo/labels'):
        os.remove('yolo/labels/'+f)
    for f in os.listdir('yolo/images'):
        os.remove('yolo/images/'+f)
    all_names = []
    for file in all_data:
        if '.zip' in file:
            with zipfile.ZipFile(base_path+'/'+ file, 'r') as zip_ref:
                name_list = zip_ref.namelist()

                for name in name_list:
                    if name in all_names:
                        print('file already exists', name)
                        continue
                    all_names.append(name)
                    
                    if 'labels' in name and '.txt' in name: 
                        
                        zip_ref.extract(name, 'yolo/')
                    if 'images' in name and '.' in name: 
                        zip_ref.extract(name, 'yolo/')
                    if 'classes.txt' in name:
                        zip_ref.extract(name, 'yolo/')

make_yolo_data() 

# this variable is set to True if you want to add data only for the training data
# since frames between videos are very similar, it is better to add them only to the training data

add_video_data=True
 
    

training_proportions = [0.8, ]
original_data_path = 'yolo/labels/'
original_images_folder = 'yolo/images/'
images  =  os.listdir(original_images_folder)
random.Random(7).shuffle(images)# generate the same shuffle always

names   = [f.split('.')[0] for f in  images ]
labels  = [name + '.txt' for name in names]

    # shuffle(images)
# remove  evrything  in train/labels and  train/images

for proportion in training_proportions:
        
    for f in os.listdir('datasets/train/images'):
        os.remove('datasets/train/images/'+f)
    for f in os.listdir('datasets/train/labels'):
        os.remove('datasets/train/labels/'+f)

    # remove  evryting  in test/labels and  test/images
    for f in os.listdir('datasets/test/images'):
        os.remove('datasets/test/images/'+f)
    for f in os.listdir('datasets/test/labels'):
        os.remove('datasets/test/labels/'+f)


    # remove  evryting  in val/labels and  val/images
    for f in os.listdir('datasets/val/images'):
        os.remove('datasets/val/images/'+f)
    for f in os.listdir('datasets/val/labels'):
        os.remove('datasets/val/labels/'+f) 


    train_images = images[ :int(len(images)*proportion)]
    test_images  = images[ int(len(images)*(0.95)):]
    val_images   = images[ int(len(images)*proportion):int(len(images)*(proportion +0.2))]



    print('train images', len(train_images))
    print('test images', len(test_images))
    print('val images', len(val_images))

    i=0 
    for name in train_images:

        file_name =name.rsplit('.',1)[0]+'.txt'
        # check if label and image exist
        if os.path.exists(original_data_path+file_name) and os.path.exists(original_images_folder+name):
            shutil.copy(original_data_path+file_name,'datasets/train/labels/'+file_name)
            shutil.copy(original_images_folder+name, 'datasets/train/images/'+name ) 
        else:
            print('file does not exist', name)
            continue



    for name in test_images :
        file_name =name.rsplit('.',1)[0]+'.txt'
        # check if label and image exist
        if os.path.exists(original_data_path+file_name) and os.path.exists(original_images_folder+name):
            shutil.copy(original_data_path+file_name, 'datasets/test/labels/'+file_name)
            shutil.copy(original_images_folder+name, 'datasets/test/images/'+name )
    

    for name in val_images:
        file_name =name.rsplit('.',1)[0]+'.txt'
        # check if label and image exist
        if os.path.exists(original_data_path+file_name) and os.path.exists(original_images_folder+name):
            shutil.copy(original_data_path+file_name, 'datasets/val/labels/'+file_name)
            shutil.copy(original_images_folder+name, 'datasets/val/images/'+name )


    # add  frames from videos to the training data
    if add_video_data:
        if  os.path.exists('video_data/images') and os.path.exists('video_data/labels'):

            for file in os.listdir('video_data/images'):
                shutil.copy('video_data/images/'+file, 'datasets/train/images/'+file) 
            
            for file in os.listdir('video_data/labels'):
                shutil.copy('video_data/labels/'+file, 'datasets/train/labels/'+file) 
        else:
            print('video data not added')
             
     
    img_sizes =[960,] 
    list_of_models= [
            ('yolov8n', 32), 
            # ('yolov8x', 4), 
            # ('yolov8l', 12), 
            # ('yolov8s', 24), 
        ]
    
    for m_name, batch in list_of_models: 
        for img_size in img_sizes:
            # build from YAML and transfer weights
            # Train the model
            model = YOLO(m_name+'.yaml').load(m_name+'.pt')
            # mosaic =0.1 aniqligni ancha yaxshi oshirgandi 
            model.train( data='data.yaml', batch=batch,  epochs=500,  patience=100,  imgsz=img_size, pretrained=True,  workers=8,  name='PEPSI_'+ m_name+'_'+ str(img_size)+"_"+str(int(len(images)*proportion)) , fliplr=0.0 ,  degrees=0.4, dropout=0.1, copy_paste=0.1 , mosaic=0)
            