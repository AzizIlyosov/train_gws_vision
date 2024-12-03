import pandas as pd 
import os 
from random import shuffle
import shutil
from ultralytics import YOLO 

import pandas as pd
import yaml
import os 
import pandas as pd
import yaml
from make_yolo_data_from_zip_folders import make_yolo_data



# extract from zip files
make_yolo_data()

# filtering data is used to remove some classes from the dataset

filtering_data=True
add_video_data=False


def filter_labels_by_classes():
    with open('data.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

        
    

    prev_classes  = {i:j for i,j in  enumerate(data['names'])}
    new_keys = pd.read_csv('keys.csv')
    new_dict = {j:i for i,j in enumerate(new_keys['class'].values)}

    for file in os.listdir('yolo/labels'):
        lines=[]
        with open('yolo/labels/'+file, 'r') as f:
            lines = f.read()
            lines = [l for l in lines.split('\n') if l!='']
            lines = [i.split(' ') for i in lines]
            # print('this is [rev classes]', prev_classes)
            # print('lines ', lines )
            lines = [i for i in lines if prev_classes[int(i[0])] in new_keys['class'].values]
            lines = [ [ str(new_dict[prev_classes[ int(line[0])] ]), ]+line[1:] for line in lines ]
            
        if len(lines)==0:
            os.remove('yolo/labels/'+file)
            os.remove('yolo/images/'+file.replace('txt','jpg'))
        else:
            with open('yolo/labels/'+file, 'w') as f:
                for line in lines:
                    f.write(' '.join(line)+'\n')   
    
    for file in os.listdir('video_data/labels'):
        lines=[]
        with open('video_data/labels/'+file, 'r') as f:
            lines = f.read()
            lines = [l for l in lines.split('\n') if l!='']
            lines = [i.split(' ') for i in lines] 
            lines = [i for i in lines if prev_classes[int(i[0])] in new_keys['class'].values]
            lines = [ [ str(new_dict[prev_classes[ int(line[0])] ]), ]+line[1:] for line in lines ]
            

        if len(lines)==0:
            os.remove('video_data/labels/'+file)
            os.remove('video_data/images/'+file.replace('txt','jpg'))
        else:
            with open('video_data/labels/'+file, 'w') as f:
                for line in lines:
                    f.write(' '.join(line)+'\n') 







if  filtering_data:
    filter_labels_by_classes() 
    

training_proportions = [0.75, ]
original_data_path = 'yolo/labels/'
original_images_folder = 'yolo/images/'
images  = os.listdir(original_images_folder)
shuffle(images)

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
    test_images  = images[ int(len(images)*(proportion+0.2)):]
    val_images   = images[ int(len(images)*proportion):int(len(images)*(0.95))]


    i=0 
    for name in train_images:
        # i+=1
        # if i>200: continue

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
        for file in os.listdir('video_data/images'):
            shutil.copy('video_data/images/'+file, 'datasets/train/images/'+file) 
        
        for file in os.listdir('video_data/labels'):
            shutil.copy('video_data/labels/'+file, 'datasets/train/labels/'+file) 
    
    # exit()
    
    img_sizes =[960,]


    list_of_models= [
            ('yolov8n', 32), 
            # ('yolov8x', 4), 
            # ('yolov8l', 12), 
            #('yolov8s', 24), 
        ]
    for m_name ,batch in list_of_models  : 
        for img_size in img_sizes:
            # build from YAML and transfer weights
            # Train the model
            model = YOLO(m_name+'.yaml').load(m_name+'.pt') 
            # mosaic =0.1 aniqligni ancha yaxshi oshirgandi 
            if filtering_data:
                model.train( data='data_filter.yaml', batch=batch,  epochs=500,  patience=50,  imgsz=img_size, pretrained=True,  workers=8,device=0,  name='PEPSI_'+ m_name+'_'+ str(img_size)+"_"+str(int(len(images)*proportion)) , fliplr=0.0 ,  degrees=0.4, dropout=0.1, copy_paste=0.1 )
            else:
                model.train( data='data.yaml', batch=batch,  epochs=500,  patience=50,  imgsz=img_size, pretrained=True,  workers=8,device=0,  name='PEPSI_'+ m_name+'_'+ str(img_size)+"_"+str(int(len(images)*proportion)) , fliplr=0.0 ,  degrees=0.4, dropout=0.1, copy_paste=0.1 )
            