#머신러닝 객체 생성 -> 훈련 -> 예측 -> 평가
#YOLO 라이브러리 세팅(pip)
from ultralytics import YOLO

def yolo_train():
    yaml_path = r'./yolo_setting.yaml'

    #YOLO 훈련
    result = YOLO('yolov8n.pt').train(
                data = yaml_path,
                epochs=50,
                imgsz=640,
                batch=16,
                save=True,
                device=0,
                plots=True,
                name='peach_train01')

    print('훈련 완료')

def yolo_predict(source, weight_path='../runs/detect/peach_train01/weights/best.pt'):
    #YOLO 평가
    #weight_path = './runs/detect/peach_train01/weights/best.pt'
    model = YOLO(weight_path)

    model.predict(source=source,
                  device=0,
                 save=True)