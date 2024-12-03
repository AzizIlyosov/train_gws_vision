from ultralytics import YOLO 
img_sizes =[960,]


list_of_models= [
         ('yolov8n', 32),
        # ('yolov8s', 11),
        #('yolov8m', 8),
        # ('yolov8l', 6),
        # ('yolov8x', 4),
    ]
for m_name ,batch in list_of_models  : 
    for img_size in img_sizes:
        # build from YAML and transfer weights
        # Train the model
        model = YOLO(m_name+'.yaml').load(m_name+'.pt') 
        # mosaic =0.1 aniqligni ancha yaxshi oshirgandi 
        model.train( data='data.yaml', batch=batch,  epochs=500,  patience=75,  imgsz=img_size, pretrained=True,  workers=8,device=0,  name='PEPSI_'+ m_name+ str(img_size) , fliplr=0.0 ,  degrees=0.4, dropout=0.1, copy_paste=0.1,   )
        
        # model = YOLO(m_name+'.yaml').load(m_name+'.pt')
        #model.train( data='data.yaml',batch=batch,  epochs=100,lr0=0.001, patience=50,  imgsz=img_size, pretrained=False,augment=True,  workers=8,device=0,mosaic=True, name='PEPSI_'+ m_name+ str(img_size)+'_mosaic', fliplr=0.0 )


# for m_name ,batch in list_of_models[::-1]: 
#     for img_size in img_sizes:
#         path = 'runs/detect/' + m_name+str(img_size)+'/weights/'
#         print('this is path  ', path )
#         model  =  YOLO(path+'best.pt' )
#         new_path = path+m_name+str(img_size)+'.onnx'
#         print('\n\n\nthis is  new_path  ', new_path)
#         model.export(format='onnx', device=0,  dynamic=True)
#         # /home/aziz/Desktop/obj_detection/runs/detect/yolov8n640/weights/best.pt
        
