import zipfile
import os 
def make_yolo_data(base_path = 'projects_to_check'):
    
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
if __name__ == "__main__":
    make_yolo_data()
    