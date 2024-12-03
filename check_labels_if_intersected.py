import os 

def intersection(obj1,obj2):

    # max_x0 = max(obj1[0],obj2[0])
    # max_y0 = max(obj1[1],obj2[1])

    # min_x1 = min(obj1[0]+obj1[2],obj2[0]+obj2[2])
    # min_y1 = min(obj1[1]+obj1[3],obj2[1]+obj2[3])

    
    # try:
    obj1[2] = obj1[2]+obj1[0]
    obj1[3] = obj1[3]+obj1[1]


    obj2[2] = obj2[2]+obj2[0]
    obj2[3] = obj2[3]+obj2[1]


    intersection = max(0, min(obj1[2], obj2[2]) - max(obj1[0], obj2[0])) * \
                        max(0, min(obj1[3], obj2[3]) - max(obj1[1], obj2[1]))
    union = (obj1[2] - obj1[0]) * (obj1[3] - obj1[1]) + \
            (obj2[2] - obj2[0]) * (obj2[3] - obj2[1]) - intersection
    # except:
    #     print("error ", obj1 , obj2)
    #     exit()
    
    if union==0:
        return 1

    iou = intersection / union
    
    # print('iou = ', iou)

    return iou





for file_name in os.listdir('yolo/labels/'):

    with open('yolo/labels/'+file_name, 'r') as file: 
        lines = file.read().split('\n')
        if lines[-1]=='':
            lines=lines[:-1]
        values =[ line.split(' ') for line in lines]

        # print('this is vlaues ', values)
        
        for index1 in range(0, len(values)): 
            obj1 = [float(v) for v in  values[index1][1:]]
            for index2 in range(index1, len(values)):
                if index2==index1:
                    continue
                obj2 = [float(v) for v in  values[index2][1:]]
                val = intersection(obj1, obj2)
                if val>0.3:
                    print('filename ', file_name, ' value ', val, ' ', values[index1][0], values[index2][0])