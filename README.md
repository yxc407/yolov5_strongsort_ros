# yolov5-strongsort-ros
This ROS package provides a real-time object detection and multi-object tracking pipeline by integrating YOLOv5 and StrongSORT. It supports multiple deep-learning frameworks for inference and is inspired by the YOLOv5 ROS implementation.

## Tested Environment
- **Operating System**: Ubuntu 20.04 LTS
- **ROS**: ROS Noetic
- **Python**: 3.8.10
- **CUDA / cuDNN**: 11.4 / 8.6.0
- **PyTorch / Torchvision**: 2.1.0 / 0.16.1
- **CvBridge**: 1.16.2

## Installation and Building
1. Create or switch to your ROS workspace (e.g., ~/catkin_ws/, ~/ros_ws/, etc.):
    
    ```
    cd <ros_workspace>/src
    ```
2. Clone this repository (and submodules if needed):
    
    ```
    git clone --recurse-submodules https://github.com/yxc407/yolov5_strongsort_ros.git
    ```
3. Install dependencies

    ```
    cd src/yolov5_strongsort_ros/scripts/boxmot
    pip install -r requirements.txt
    ```
4. Build the ROS package
    
    ```
    cd <ros_workspace>
    catkin_make -DCATKIN_WHITELIST="yolov5_strongsort_ros"
    ```

## Basic usage
This package provides two main ROS launch files:

Change the parameter for input_image_topic in launch/detector.launch to any ROS topic with message type of sensor_msgs/Image. Other parameters can be modified or used as is.
- **detector.launch**

    Performs only object detection using YOLOv5.

    ```
    roslaunch yolov5_strongsort_ros detector.launch
    ```

- **tracker.launch**

    Performs both object detection and multi-object tracking.

    ```
    roslaunch yolov5_strongsort_ros tracker.launch
    ```

- **Using Messages in C++**

    In your C++ code, you can subscribe to these topics and use the provided message definitions by including the respective headers. For example:
    ```
    #include <yolov5_strongsort_ros/BoundingBox.h>
    #include <yolov5_strongsort_ros/BoundingBoxes.h>
    #include <yolov5_strongsort_ros/TrackedObject.h>
    #include <yolov5_strongsort_ros/TrackedObjects.h>

    // Example subscriber callback:
    void detectionCallback(const yolov5_strongsort_ros::BoundingBoxes::ConstPtr& msg) {
        // Process detection results
    }

    void trackingCallback(const yolov5_strongsort_ros::TrackedObjects::ConstPtr& msg) {
        // Process tracking results
    }
    ```

## Using custom weights and dataset (Working)

## References
Special thanks to the developers of the following packages:
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [BoxMot](https://github.com/mikel-brostrom/boxmot.git)
- [YOLOv5 ROS](https://github.com/mats-robotics/yolov5_ros.git)