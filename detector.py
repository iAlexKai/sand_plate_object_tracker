"""detector.py
This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""

import sys
import argparse
import cv2
import socket
import pycuda.autoinit  # This is needed for initializing CUDA driver

from collections import defaultdict
from utils.yolo_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3
from utils.camera import add_camera_args, Camera
from utils.visualization import open_window, show_fps, record_time, show_runtime
from utils.engine import BBoxVisualization


WINDOW_NAME = 'TensorRT YOLOv3 Detector'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v2_coco'
]
class_path='config/classes_5.names'


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().strip().split("\n")
    print("Loaded class {}".format(",".join(names)))
    return names

classes = load_classes(class_path)

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Family')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                 'yolov3-tiny-288', 'yolov3-tiny-416', 'yolov3-custom-tiny-416'])
    parser.add_argument('--runtime', action='store_true',
                        help='display detailed runtime')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, runtime, trt_yolov3, conf_th, vis, s):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      s: socket
    """

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        timer = cv2.getTickCount()
        img = cam.read().copy()
        if img is not None:
            width = img.shape[1]
            height = img.shape[0]
            all_positions = defaultdict(list)
            if runtime:
                boxes, confs, clss, _preprocess_time, _postprocess_time, _network_time = trt_yolov3.detect(img, conf_th)
                assert len(boxes) == len(clss)
                for i in range(len(boxes)):
                    cls = classes[int(clss[i])]
                    x1, y1, x2, y2 = boxes[i]
                    middle_x = (x1 + x2) / 2
                    # middle_y = (y1 + y2) / 2
                    all_positions[cls].append([1, middle_x / width, y1 / height])
                # import pdb
                # pdb.set_trace()
                img, _visualize_time = vis.draw_bboxes(img, boxes, confs, clss)
                time_stamp = record_time(_preprocess_time, _postprocess_time, _network_time, _visualize_time)
                show_runtime(time_stamp)
            else:
                boxes, confs, clss, _, _, _ = trt_yolov3.detect(img, conf_th)
                img, _ = vis.draw_bboxes(img, boxes, confs, clss)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            img = show_fps(img, fps)

            current_socket, addr = s.accept()  # 建立客户端连接
            current_socket.send(bytes(str(dict(all_positions)), encoding='utf-8'))
            current_socket.close()
            
            cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_socket():
    def _get_local_ip():
        csock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csock.connect(('8.8.8.8', 80))
        addr, _ = csock.getsockname()
        return addr
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
    # host = "10.105.240.42"  # 获取本地主机名，注意，当前的本机就是主楼那台服务器
    host = _get_local_ip()  # 获取本地主机名，注意，当前的本机就是主楼那台服务器
    port = 2222  # 设置端口
    print("{} {} waiting for connection".format(host, port))
    s.bind((host, port))  # 绑定端口
    s.listen(5)  # 等待客户端连接
    return s


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('[INFO]  Failed to open camera!')

    # cls_dict = get_cls_dict('coco')
    cls_dict = {0: 'fire', 1: 'amb', 2: 'bus', 3: 'oil', 4: 'lug'}
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
    trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim), category_num=5)

    print('[INFO]  Camera: starting')
    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'TensorRT YOLOv3 Detector')
    vis = BBoxVisualization(cls_dict)

    s = get_socket()
    loop_and_detect(cam, args.runtime, trt_yolov3, conf_th=0.3, vis=vis, s=s)

    print('[INFO]  Program: stopped')
    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
