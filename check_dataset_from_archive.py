import cv2 
import os 
import zipfile 
import numpy as np 
import argparse
import shutil 
import os, zipfile

folder_names = [f.split('.')[0] for f in  os.listdir('projects_to_check')]

for folder_name in folder_names:

    if not(os.path.exists('check_dataset/'+folder_name + '_check_dataset' )):
        os.mkdir('check_dataset/'+folder_name+'_check_dataset' ) 

    f=zipfile.ZipFile('projects_to_check/'+folder_name+'.zip', 'r')

    all_image_names = []
    
    classes = [cl  for cl in f.read('classes.txt').decode('utf-8').split('\n') if not(cl=='' or cl==' ')]
    

    if not 'images/' in f.namelist():
        print('wrong format of the data') 
    else:
        for file_name in f.namelist():
            if (file_name!='classes.txt' and '.' in file_name and 'images/' in file_name) :
                f_name = file_name.split('/')[1]
                all_image_names.append(f_name)
            else :
                print('this is filename ', file_name)

    for file_name in all_image_names:
        img_file = f.read('images/'+file_name) 
        image = cv2.imdecode(np.frombuffer(img_file, np.uint8), 1)  
        shape  = image.shape[:2]

        txt = f.read('labels/'+file_name.rsplit('.',1)[0]+'.txt' ).decode('utf-8') 
        lines = str(txt).split('\n') 
        for idx, line in enumerate(lines):
            if line=='' or  line==' ':
                continue
            line   = line.split()
            class_ = classes[int(line[0])]
            coords = [float(i) for i in line[1:]]

            coords[0] = int(coords[0]*shape[1])-int(coords[2]* shape[1])//2
            coords[1] = int(coords[1]*shape[0])-int(coords[3]* shape[0])//2
            coords[2] = int(coords[2]*shape[1])
            coords[3] = int(coords[3]*shape[0])

            if not(os.path.exists('check_dataset/'+folder_name+ '_check_dataset/'+str(class_))):
                os.mkdir('check_dataset/'+folder_name+'_check_dataset/'+str(class_)) 
            cut_image = image[ coords[1]:coords[1]+coords[3],coords[0]: coords[0]+coords[2] ]
            try:
                cv2.imwrite( 'check_dataset/'+folder_name+ '_check_dataset/'+str(class_) +'/'+ file_name+'_'+str(idx)+'.jpg', cut_image)
            except Exception as ex:
                print('check_dataset/'+folder_name +'_check_dataset/'+str(class_) +'/'+ file_name+'_'+str(idx)+'   '+  str(ex)+ '    ', coords) 

    name = 'check_dataset/'+folder_name+ '_check_dataset/'
    zip_name = name + '.zip'

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(name):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))
    zip_ref.close()

