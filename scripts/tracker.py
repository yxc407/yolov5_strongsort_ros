#!/usr/bin/env python3

import os
import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from yolov5_strongsort_ros.msg import TrackedObjects, TrackedObject

from pathlib import Path
import sys
import threading
import queue

# limit the number of CPUs used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# add yolov5 and strong_sort submodules to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'boxmot'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 and strong_sort submodules
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (
    check_img_size, 
    non_max_suppression, 
    scale_coords,
    LOGGER, 
    xyxy2xywh
)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.augmentations import letterbox
from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config

class StrongsortTracker:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.max_det = rospy.get_param("~maximum_detections", 20)
        self.view_image = rospy.get_param("~view_image", False)
        # Initialize weights 
        self.yolo_weights = Path(rospy.get_param("~yolo_weights"))
        self.strongsort_weights = Path(rospy.get_param("~strong_sort_weights"))
        self.device_str = rospy.get_param("~device")

        cfg = get_config()
        cfg.merge_from_file(rospy.get_param("~config", Path(__file__).parent / "boxmot/strong_sort/configs/strong_sort.yaml"))
        
        self.source_topic = rospy.get_param("~input_image_topic")
        self.pub_topic = rospy.get_param("~output_topic")
        self.publish_image = rospy.get_param("~publish_image")
        self.pub_image_topic = rospy.get_param("~output_image_topic")
        
        # Initialize device
        self.device = select_device(self.device_str)
        rospy.loginfo(f"Using device: {self.device}")

        # Enable CuDNN benchmark for optimized performance
        torch.backends.cudnn.benchmark = True
        
        # Initialize model
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device)
        self.model.eval()  # Set model to evaluation mode
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.names = names
        self.img_size = [rospy.get_param("~inference_size_w", 320), rospy.get_param("~inference_size_h", 320)]
        self.img_size = check_img_size(self.img_size, s=stride)
        rospy.loginfo(f"Model image size: {self.img_size}")
        
        # Half
        self.half = rospy.get_param("~half", False)

        self.strongsort = StrongSORT(
            self.strongsort_weights,
            self.device,
            fp16 = self.half,
            max_dist = cfg.STRONGSORT.MAX_DIST,
            max_iou_distance = cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age = cfg.STRONGSORT.MAX_AGE,
            n_init = cfg.STRONGSORT.N_INIT,
            nn_budget = cfg.STRONGSORT.NN_BUDGET,
            mc_lambda = cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha = cfg.STRONGSORT.EMA_ALPHA,
        )
        self.strongsort.model.warmup()

        # Warmup YOLO model
        self.model.warmup(imgsz=(1, 3, *self.img_size))
        
        # Initialize ROS Publisher / Subscriber
        self.bridge = CvBridge()

        # Determine if the input topic is CompressedImage or Image
        try:
            input_image_type, _, _ = rospy.get_topic_type(self.source_topic, blocking=True)
        except:
            input_image_type = "sensor_msgs/Image"  # default to Image if unable to get type

        if input_image_type == "sensor_msgs/CompressedImage":
            self.image_sub = rospy.Subscriber(
                self.source_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24
            )
        else:
            self.image_sub = rospy.Subscriber(
                self.source_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
            )

        # Initialize prediction publisher
        self.tracks_pub = rospy.Publisher(self.pub_topic, TrackedObjects, queue_size=1)

        # Initialize image publisher
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                self.pub_image_topic, Image, queue_size=1
            )

        # Initialize processing queue and thread
        self.frame_queue = queue.Queue(maxsize=1)
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            if isinstance(msg, CompressedImage):
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Try to put the frame into the queue. If the queue is full, drop the frame to reduce latency.
            if not self.frame_queue.full():
                self.frame_queue.put((frame, msg.header))
            # else:
            #     rospy.logwarn("Frame queue is full. Dropping frame to reduce latency.")
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def process_frames(self):
        while not rospy.is_shutdown():
            try:
                frame, header = self.frame_queue.get(timeout=1)
                tracked_objects, annotated_frame = self.run_tracker(frame)
                
                # Create TrackedObjects message
                to_msg = TrackedObjects()
                to_msg.header = header
                to_msg.image_header = header

                for t in tracked_objects:
                    obj_msg = TrackedObject()
                    obj_msg.track_id = t["track_id"]
                    obj_msg.Class = t["class_name"]
                    obj_msg.probability = t["conf"]
                    obj_msg.xmin = t["xmin"]
                    obj_msg.ymin = t["ymin"]
                    obj_msg.xmax = t["xmax"]
                    obj_msg.ymax = t["ymax"]
                    to_msg.tracked_objects.append(obj_msg)
                
                # Publish tracked objects
                self.tracks_pub.publish(to_msg)

                # Publish annotated image
                if self.publish_image:
                    out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                    self.image_pub.publish(out_msg)

                # Show video if enabled
                if self.view_image:
                    cv2.imshow('tracking', annotated_frame)
                    cv2.waitKey(1)
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error in process_frames: {e}")

    def run_tracker(self, frame):
        im0 = frame.copy()

        img, ratio, pad = letterbox(frame, new_shape=self.img_size, auto=True)
        
        # Convert BGR to RGB, transpose, and normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.float() / 255.0
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)  # (1, 3, h, w)

        # YOLO inference
        with torch.no_grad():
            pred = self.model(img_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)[0]
        
        det = pred.cpu() if pred is not None else None

        annotator = Annotator(im0, line_width=2, pil=False)
        tracked_objects = []

        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape, ratio_pad=(ratio, pad)
            ).round()

            # Check for valid detections
            widths = det[:, 2] - det[:, 0]
            heights = det[:, 3] - det[:, 1]
            valid_indices = (widths > 0) & (heights > 0)
            det = det[valid_indices]

            if len(det) == 0:
                self.strongsort.increment_ages()
                return [], annotator.result()

            # Prepare inputs for StrongSORT
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5].long()  # Ensure clss is long

            # Tracking
            outputs = self.strongsort.update(xywhs, confs, clss, im0)

            if len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_id, cls_id, conf = output
                    class_id = int(cls_id)
                    name = self.names[class_id] if class_id < len(self.names) else "unknown"
                    color = colors(class_id, True)
                    label = f"ID:{int(track_id)} {name} {conf:.2f}"
                    annotator.box_label([x1, y1, x2, y2], label, color=color)

                    # Collect tracking data
                    tracked_objects.append({
                        "track_id": int(track_id),
                        "class_id": class_id,
                        "class_name": name,
                        "conf": float(conf),
                        "xmin": int(x1),
                        "ymin": int(y1),
                        "xmax": int(x2),
                        "ymax": int(y2)
                    })
            
        else:
            # If no detections, increment ages in tracker
            self.strongsort.increment_ages()

        # Annotate the image
        im0 = annotator.result()

        return tracked_objects, im0

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node('tracker', anonymous=True)
    tracker = StrongsortTracker()

    tracker.spin()
