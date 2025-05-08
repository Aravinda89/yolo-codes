from ultralytics import YOLO

model = YOLO('./models/yolov11_traffic_lights.pt')

# model.predict(source='./data/traffic_light/test/images/no-brand_no-brand_full01-1-_jpg.rf.cb7cd75ce141385297b131c06fea7013.jpg',
#              show=True, save=True, conf=0.6, line_width=2,save_crop=True, save_txt=True, show_labels=True, show_conf =True, classes=[0,1])

model.predict(source='./data/traffic.mp4',show=True, save=True, conf=0.7, line_width=2, save_crop=False, save_txt=False, show_labels=True, show_conf =True)

# local webcam source='0'
