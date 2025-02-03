# yolov5_strongsort_ros

This ROS package provides a real-time object detection and multi-object tracking pipeline by integrating YOLOv5 and StrongSORT. It supports multiple deep-learning frameworks for inference and is inspired by the YOLOv5 ROS implementation.

## Tested Environment
- **Operating System**: Ubuntu 20.04 LTS
- **ROS**: ROS Noetic
- **Python**: 3.8.10
- **CUDA / cuDNN**: 11.4 / 8.6.0
- **PyTorch / Torchvision**: 2.1.0 / 0.16.1
- **CvBridge**: 1.16.2

## Installation and Building
1. **Create or Switch to Your ROS Workspace**
    
    ```bash
    cd <ros_workspace>/src
    ```
2. **Clone This Repository**
    
    ```bash
    git clone --recurse-submodules https://github.com/yxc407/yolov5_strongsort_ros.git
    ```
3. **Install Dependencies**
    
    ```bash
    cd yolov5_strongsort_ros/scripts/boxmot
    pip install -r requirements.txt
    ```
4. **Build the ROS Package**
    
    ```bash
    cd <ros_workspace>
    catkin_make -DCATKIN_WHITELIST="yolov5_strongsort_ros"
    ```

## Basic Usage

This package provides two main ROS launch files:

**Before launching, modify the `input_image_topic` parameter in `launch/detector.launch` to subscribe to the desired ROS topic that publishes messages of type `sensor_msgs/Image`. Other parameters can be modified as needed or used with their default values.**

- **detector.launch**
  
    Performs only object detection using YOLOv5.
  
    ```bash
    roslaunch yolov5_strongsort_ros detector.launch
    ```
  
- **tracker.launch**
  
    Performs both object detection and multi-object tracking.
  
    ```bash
    roslaunch yolov5_strongsort_ros tracker.launch
    ```

- **Using Messages in C++**
  
    In your C++ code, you can subscribe to these topics and use the provided message definitions by including the respective headers. For example:
    ```c++
    #include <yolov5_strongsort_ros/BoundingBox.h>
    #include <yolov5_strongsort_ros/BoundingBoxes.h>
    #include <yolov5_strongsort_ros/TrackedObject.h>
    #include <yolov5_strongsort_ros/TrackedObjects.h>
    ```

## Training and Using Custom Weights

### YOLOv5 Training

1. **Organize Your Dataset**
    
    Ensure your dataset is organized in the YOLO format within the `yolov5` directory. The directory structure should resemble the following:
    ```bash
    yolov5/
    ├── datasets/
    │   └── your_dataset/
    │       ├── images/
    │       │   ├── train/
    │       │   └── val/
    │       └── labels/
    │           ├── train/
    │           └── val/
    ```
2. **Create a YAML Configuration File**
    
    Create a YAML file (e.g., `coco_PersonOnly.yaml`) inside the `data/` directory to define your dataset paths and classes:
    ```yaml
    train: datasets/your_dataset/images/train
    val: datasets/your_dataset/images/val

    nc: 1
    names: ['person']
    ```
3. **Train the YOLOv5 Model**
    
    Run the training command in the `yolov5` root directory:
    ```bash
    python train.py --img 320 --batch 8 --epochs 50 --data data/coco_PersonOnly.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name your_training_name
    ```
    - `--img 320`: Sets the image size to 320x320 pixels.
    - `--batch 8`: Sets the batch size to 8.
    - `--epochs 50`: Trains the model for 50 epochs.
    - `--data`: Path to your dataset YAML file.
    - `--cfg`: Model configuration file.
    - `--weights`: Pre-trained weights to start from.
    - `--name`: Name for the training run.
4. **Monitor Training**
    
    The training process will display logs and save checkpoints in the `runs/train/your_training_name/` directory.

### Using Custom Weights

1. **Create Weights Directory**
    
    Create a `weights/` directory in the `yolov5` root:
    ```bash
    mkdir yolov5/weights
    ```
2. **Move Trained Weights**
    
    Move the trained weights (`best.pt` or `last.pt`) to the `weights/` directory and rename them appropriately:
    ```bash
    mv runs/train/your_training_name/weights/best.pt yolov5/weights/your_model_name.pt
    ```
3. **Modify Paths in `detector.launch`**
    
    Open the `detector.launch` file and update the paths for the weights and data arguments:
    ```xml
    <arg name="weights" default="$(find yolov5_strongsort_ros)/scripts/boxmot/yolov5/weights/your_model_name.pt" />
    <arg name="data" default="$(find yolov5_strongsort_ros)/scripts/boxmot/yolov5/data/coco_PersonOnly.yaml" />
    ```
    Ensure that the paths correctly point to the newly trained weights and dataset configuration.

## Troubleshooting
1. No module named 'torchreid'
    ```bash
    pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip
    ```
    See: https://github.com/phil-bergmann/tracking_wo_bnw/issues/159

## References

Special thanks to the developers of the following packages:
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [BoxMot](https://github.com/mikel-brostrom/boxmot)
- [YOLOv5 ROS](https://github.com/mats-robotics/yolov5_ros)
