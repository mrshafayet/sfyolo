from ultralytics.models import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    model = YOLO(model='./run/train/exp3/weights/best.pt')
    model.val(data='./datasets/data.yaml', split='val', batch=192, device='0', project='run/val', name='exp',
              half=False,)
