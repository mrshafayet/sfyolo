# SFYOLO: A Customized YOLO Implementation

SFYOLO is a custom implementation of the YOLO (You Only Look Once) object detection algorithm. It incorporates modular improvements and enhancements designed to optimize performance and flexibility for various computer vision tasks, such as object detection, segmentation, and classification.

## Table of Contents
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Dataset YAML](#dataset-yaml)
- [Models and Performance](#models-and-performance)
- [License](#license)

## Features



## Installation
### Prerequisites
- Python 3.8 or higher
- CUDA (if using GPU acceleration)
- PyTorch 1.11 or higher

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/mrshafayet/sfyolo.git
   cd sfyolo
   ```

3. Set up datasets:
   - Place datasets in the `datasets/` directory.
   - Update the configuration files in `ultralytics/cfg/` as needed.
   ## Dataset YAML
### SIMD Dataset
Setup the SIMD Dataset or your training dataset yaml in train.py
The SIMD dataset configuration can be accessed using the following link: [SIMD Dataset YAML](https://github.com/mrshafayet/sfyolo/datasets/simd.yaml)

Place the dataset files in the `datasets/` directory and ensure the paths in the YAML file are updated correctly.

### SIMD Dataset Download
The dataset can be downloaded using the following Baidu Drive link: [Satellite Imagery Multi-vehicles Dataset (SIMD)](https://pan.baidu.com/s/1GdV14rd2BwxZ3gwIKni4hQ) (Password: 1234)

## Usage
### Training
To train a model, use the following command:
```bash
python ultralytics/engine/train.py --cfg ultralytics/cfg/xy_l.yaml --epochs 150 --data datasets/config.yaml
```

### Evaluation
Evaluate the model on a validation set:
```bash
python ultralytics/engine/val.py --weights run/weights/xy_l.pt --data datasets/config.yaml
```

### Inference
Run inference on a single image or a folder of images:
```bash
python ultralytics/engine/predict.py --weights run/weights/xy_l.pt --source input.jpg
```



## Models and Performance
| **Model** | **Parameters (M)** | **GFLOPs** | **mAP@50 (%)** |
|-----------|---------------------|------------|----------------|
| **xy_n**  | 2.25               | 8.7        | 73.4           |
| **xy_s**  | 8.08               | 29.8       | 79.2           |
| **xy_m**  | 20.37              | 103.0      | 82.1           |
| **xy_l**  | 23.65              | 124.5      | 81.6           |
| **xy_x**  | 53.13              | 278.9      | 82.0           |

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or issues, please open an issue on the [GitHub repository](https://github.com/mrshafayet/sfyolo/issues).
