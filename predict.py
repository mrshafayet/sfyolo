from ultralytics.models import YOLO


if __name__ == '__main__':
    model = YOLO(model='run/train/exp7/weights/best.pt')
    model.predict(source='/Users/shafayettajrian/Desktop/datasets/SIMD/images/val', device='CPU', imgsz=640, project='run/detect/', name='exp',save=True)
