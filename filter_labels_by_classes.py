# this fil is used to filter all usless classes from the dataset

import os 
import pandas as pd
import yaml
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

# make filter data 
data_filter = data.copy()

data_filter['names'] = list(new_keys['class'].values)


print(data_filter['names'])

data_filter['nc'] = len(data_filter['names'])
#save  data_filter

with open('data_filter.yaml', 'w') as f:
    yaml.dump(data_filter, f)   


    

         
