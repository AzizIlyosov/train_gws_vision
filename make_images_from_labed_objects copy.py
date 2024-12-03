from flask  import Flask , request,jsonify,url_for,Response
import json 
from label_studio import webhooks
from label_studio_sdk import Client
from label_studio_sdk.project import ProjectSampling
import  cv2  
import json  
import requests
import zipfile
import numpy  as  np
import base64
import datetime 
from multiprocessing import Process 
import  torch_lightning_model
import os   
# from task_data_gen import  torch_dataset_by_json


LABEL_STUDIO_URL = 'http://192.168.40.27:8080'
LABEL_STUDIO_API_KEY = '0da6b117a50c9eea5171c362e0a1ea3f339d62fa'
ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)


# name_of_output_folder = 'output_no_label'
# if name_of_output_folder not in os.listdir():
#     os.mkdir(name_of_output_folder)


def get_labels(project_id): 
    c=0
    my_project = ls.get_project(project_id) 
    # print('request.base_url', request.host_url)
    # print(safe_join(request.base_url, 'static'))
    # train()
    # print('this is base url ', request.base_url)
    tasks   = my_project.get_labeled_tasks() 
    labels =  []   
    try:
        for  task in tasks :  
            if  task['annotations'][0]['was_cancelled']   :
                # print('task was cancelled')
                c+=1
                print(c)
                # exit()
                continue
            filename = task['storage_filename'] 
            r = task['annotations'][0]['result']
            len_labels = len(r)
            for  i in range(len_labels): 
                s= {}
                if 'rectanglelabels' not in r[i]['value'] or len( r[i]['value'])==0:
                    s['filename'] = filename
                    s['rect'] = r[i]['value'] 
                    s['label'] = "no_label"
                    s['original_width'] = r[i]['original_width']
                    s['original_height'] = r[i]['original_height']
                    labels.append(s)
                else:
                    s['filename'] = filename
                    s['rect'] = r[i]['value']
                    s['label'] = r[i]['value']['rectanglelabels'][0]
                    s['original_width'] = r[i]['original_width']
                    s['original_height'] = r[i]['original_height']
                    labels.append(s) 
    except Exception as ex:
        # print
        print('here are exc ', str(ex), r[i]) 
            

    return labels

projets= {

        # 'pepsi_1_1' : 239,
        # 'pepsi_1_2' : 240,
        # 'pepsi_1_3' : 241,
        # 'pepsi_1_4' : 242,
        # 'pepsi_1_5' : 243,
        # 'pepsi_1_6' : 245, 
        # 'pepsi_2_1' : 246,
        # 'pepsi_2_2' : 248,
        # 'pepsi_2_3' : 249,    
        # 'pepsi_2_4' : 250, 
        # 'pepsi_2_5' : 251, 
        # 'pepsi_2_6' : 252, 

        'pepsi_3_1' : 256,
        'pepsi_3_2' : 257,
        'project_yolo':272,
        'pepsi_4_0':273,
        'pepsi_4_1':274,
        'pepsi_4_2':275,
}

base_path        = 'not_labeled_objects/projects/'
class_base_path_train =  'not_labeled_objects/classes/train'
class_base_path_test =  'not_labeled_objects/classes/test'

def make_images_from_labed_objects():
    for proj in projets:

        if proj not in os.listdir(base_path):
            os.mkdir(base_path+'/'+proj)
        
        all_labels =  get_labels(projets[proj])
        print('all labels ', all_labels)

        json.dump(all_labels, open('labels.json', 'w'))
        s = ''
        for k,l in enumerate(all_labels):
            # print('this is the file ', l['filename'])
            # with open(l['filename']) as f:
            #     # print('this is the file ', f)
            img = cv2.imread(l['filename'])
            x,y,width, height = int(l['rect']['x'] * l['original_width']/100) , int(l['rect']['y']* l['original_height']/100), int(l['rect']['width']* l['original_width']/100), int(l['rect']['height']* l['original_height']/100)
            # print('this is the rect ', x,y,width, height)
            cut = img[y:y+height, x:x+width] 
            name = l['filename'].split('/')[-1].split('.')[0] + '_'+str(k)+'.jpg'
            if not os.path.exists(base_path+'/'+proj+'/'+l['label'] ):
                os.mkdir(base_path+'/'+proj+'/'+l['label'])  

            s+=name+'\n'
            try:
                if height>10  and width>3:
                    cv2.imwrite(base_path+'/'+proj+'/'+l['label']+'/'+name, cut) 
            except Exception as ex:
                print('there is error file', name )
        print('done')

make_images_from_labed_objects()


# copy file for training data

import shutil  
from random import shuffle
if not os.path.exists(class_base_path_train):
    os.mkdir(class_base_path_train)


projects = os.listdir(base_path)
for proj in projects: 
    folders = os.listdir(base_path+'/'+proj)
    for folder in folders:
        files = os.listdir(base_path+'/'+proj+'/'+folder)
        shuffle(files)
        train_files = files[:int(0.75*len(files))]
        test_files = files[int(0.75*len(files)):]
        for file in train_files:
            # print(folder , classs_base_path)
            # print('this is the file ', classs_base_path+'/'+folder)
            if not os.path.exists(class_base_path_train+'/'+folder):
                os.mkdir(class_base_path_train+'/'+folder)
            shutil.copy(base_path+'/'+proj+'/'+folder+'/'+file, class_base_path_train+'/'+folder)
        
        for file in test_files:
            # print(folder , classs_base_path)
            # print('this is the file ', classs_base_path+'/'+folder)
            if not os.path.exists(class_base_path_test+'/'+folder):
                os.mkdir(class_base_path_test+'/'+folder)
            shutil.copy(base_path+'/'+proj+'/'+folder+'/'+file, class_base_path_test+'/'+folder)

        
