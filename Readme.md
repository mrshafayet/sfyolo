# SFYOLO: A Customized YOLO Implementation

SFYOLO is a custom implementation of the YOLO (You Only Look Once) object detection algorithm. It incorporates modular improvements and enhancements designed to optimize performance and flexibility for various computer vision tasks, such as object detection, segmentation, and classification.

## Installation

### Prerequisites
- CPU: Intel 12900KF
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





## Models and Performance

| **Model** | **Parameters (M)** | **GFLOPs** | **mAP@50 (%)** | **Weights** |
|-----------|---------------------|------------|----------------|-------------|
| **xy_n**  | 2.25               | 8.7        | 73.4           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_n%2073.4/weights/best.pt) |
| **xy_s**  | 8.08               | 29.8       | 79.2           |  |
| **xy_m**  | 20.37              | 103.0      | 82.1           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_m%2081.4/weights/best.pt) |
| **xy_l**  | 23.65              | 124.5      | 81.6           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_l%20%2081/weights/best.pt) |
| **xy_x**  | 53.13              | 278.9      | 82.0           | [Download](https://github.com/mrshafayet/sfyolo/blob/main/run/train/xy_x%2081.8/weights/best.pt) |

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or issues, please open an issue on the [GitHub repository](https://github.com/mrshafayet/sfyolo/issues).
