# SFYOLO: A Customized YOLO Implementation

The YOLO network comprises three primary components: Backbone, Neck, and Head. It employs picture preprocessing, model forward propagation, and post-processing. The backbone network comprises the Conv module, SF Model module, and SPPF_WD module. GhostConv is a lightweight convolutional module that consolidates information features across channels and produces new feature maps. The SF Block module integrates the ResNet concept with a spatial attention mechanism, directing the network to concentrate on essential elements. SPPF_WD is a methodology employed in deep learning and computer vision, particularly in object detection applications. LightConv is a streamlined convolution technique that minimizes computational complexity and parameterization in models. The SF Model and SPPF_WD are modular components that accommodate feature maps of varying dimensions from the trunk and execute dimensional alignment with the intermediate scale feature layer. The RepVGG model is a streamlined VGG architecture that extensively employs 3x3 convolutions, batch normalization layers, ReLU activation functions, and significant parameterization to enhance performance.

### Overall Structure of the Model

The overall architecture of the SFYOLO model is depicted below:  
[Model Architecture](https://github.com/mrshafayet/sfyolo/blob/main/sfyolo_a.jpg)  
![Model Architecture](https://github.com/mrshafayet/sfyolo/blob/main/sfyolo_a.jpg)

## Installation

### Prerequisites
- GPU: NVIDIA 3090
- PyTorch 1.13.0
- Python 3.9

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/mrshafayet/sfyolo.git
   cd sfyolo
   ```


2. Set up datasets:
   - Place datasets in the `datasets/` directory.
   - Update the configuration files in ``train.py`` as needed.

## Dataset YAML

### SIMD Dataset YAML
Setup the SIMD Dataset or your training dataset YAML file in `train.py`.
The SIMD dataset configuration can be accessed using the following link: [SIMD Dataset YAML](https://github.com/mrshafayet/sfyolo/blob/main/datasets/SIMD.yaml).

### SIMD Dataset Download
The dataset can be downloaded using the following Baidu Drive link: [Satellite Imagery Multi-vehicles Dataset (SIMD)](https://pan.baidu.com/s/1GdV14rd2BwxZ3gwIKni4hQ) (Password: 1234).

Place the dataset files in the `datasets/` directory and ensure the paths in the YAML file are updated correctly.

## Usage

## Model Configuration

### 1SF_yolov1.yaml
The `1SF_yolov1.yaml` file defines the architecture and settings for the `SF-YOLO`  model.

#### Direct File Access
Access the file here: [1SF_yolov1.yaml](https://github.com/mrshafayet/sfyolo/blob/main/ultralytics/cfg/models/xy_YOLO/1SF_yolov1.yaml).
### Training
To train the model, you can use different configurations by updating the `train.py` file as shown below:

```python
from ultralytics.models import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/xy_YOLO/1SF_yolov1n.yaml')
    # Uncomment the desired model configuration:
    # model = YOLO(model='ultralytics/cfg/models/xy_YOLO/1SF_yolov1m.yaml')
    # model = YOLO(model='ultralytics/cfg/models/xy_YOLO/1SF_yolov1s.yaml')
    # model = YOLO(model='ultralytics/cfg/models/xy_YOLO/1SF_yolov1x.yaml')

    model.train(data='datasets/SIMD.yaml', epochs=100, batch=8, device='cpu', imgsz=640, workers=1, cache=False,
                amp=True, mosaic=False, project='run/train', name='exp')

# To monitor training, use TensorBoard:
# tensorboard --logdir=./
```

Run the training command:
![Training Process](https://github.com/mrshafayet/sfyolo/blob/main/train_model.png)
## Models and Performance
- [Pre-Trained weight of our Lightweight Model](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_n%2073.4/weights/best.pt) - 150 epoch
- [Pre-Trained weight of our Best Model](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_x%2081.8/weights/best.pt) - 150 epoch
 
| **Model** | **Parameters (M)** | **GFLOPs** | **mAP@50 (%)** | **Weights** |
|-----------|---------------------|------------|----------------|-------------|
| **xy_n(Lightweight)**  | 2.25               | 8.7        | 73.4           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_n%2073.4/weights/best.pt) |
| **xy_s**  | 8.08               | 29.8       | 79.2           |  |
| **xy_m**  | 20.37              | 103.0      | 82.1           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_m%2081.4/weights/best.pt) |
| **xy_l**  | 23.65              | 124.5      | 81.6           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_l%20%2081/weights/best.pt) |
| **xy_x(Best)**  | 53.13              | 278.9      | 82.0           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_x%2081.8/weights/best.pt) |

## mAP Graph of the Models

The following graph represents the mAP50 and mAP50-95 (mean Average Precision) performance of the models evaluated in this project:

![mAP Graph](https://github.com/mrshafayet/sfyolo/blob/main/mAP%20Graphs.png)

This analysis was performed using TensorFlow. The graph provides a clear visualization of the model performance across different configurations or epochs.

### Model Predictions

Below are the predictions from different versions of the models evaluated in this project:

- **Model xy_l**  
  [Validation Batch Labels](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_l%20%2081/val_batch2_labels.jpg)  
  ![xy_l Labels](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_l%20%2081/val_batch2_labels.jpg)

- **Model xy_m**  
  [Validation Batch Predictions](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_m%2081.4/val_batch1_pred.jpg)  
  ![xy_m Predictions](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_m%2081.4/val_batch1_pred.jpg)

- **Model xy_n**  
  [Validation Batch Predictions](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_n%2073.4/val_batch2_pred.jpg)  
  ![xy_n Predictions](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_n%2073.4/val_batch2_pred.jpg)

- **Model xy_x**  
  [Validation Batch Predictions](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_x%2081.8/val_batch2_pred.jpg)  
  ![xy_x Predictions](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_x%2081.8/val_batch2_pred.jpg)

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex

@InProceedings{
    author    = {Tajrian ABM Shafayet and Hiba Maryam},
    title     = {SFYOLO},
    booktitle = {2025 IEEE XX International Conference on XX},
    month     = {January},
    year      = {2025},
    organization = {Huazhong University of Science and Technology},
    address   = {Wuhan, China},
    publisher = {IEEE},
    note      = {Available at [insert link or DOI if applicable]}
}

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or issues, please open an issue on the [GitHub repository](https://github.com/mrshafayet/sfyolo/issues).
