
本项目使用trt压缩后的darknet模型，可以达到20的fps

1. 通过服务器上/home/myk/yolo/PyTorch-YOLOv3工程训练出来yolo-tiny模型后，通过model.save_darknet_weights方法将checkpoints中的pt权重文件转化成为weights文件
这是因为TRT-yolov3工程仅支持weights权重文件的trt压缩
转化过程参考https://zhuanlan.zhihu.com/p/143365073


2. 将转化后的weights文件从服务器上拷贝到 /home/nano/Desktop/workspace_myk/TRT-yolov3/yolov3_onnx目录下
    先后使用yolov3_to_onnx.py和onnx_to_tensorrt.py两个脚本将模型转为trt格式
3. 修改utils/yolov3.py文件中的output_shapes和category_num，与自己训练的模型中检测的类别数对应上。
    参考：https://github.com/jkjung-avt/tensorrt_demos/issues/55
4. 在TRT-yolov3目录下运行
    python3 detector.py --file --filename ./data/videos/2.mp4 --model yolov3-custom-tiny-416 --runtime
    即可达到20fps的检测效果



与服务器上基于pytorch的不同，这个完全是基于tensorrt

摄像头的帧率应该是60hz
不经过tensorrt转换直接使用pytorch在服务器上可以达到 50fps左右的帧率

不经过tensorrt转换后在nano上最多只能达到1到2fps
经过tensorrt转换后在nano上大概能达到20fps左右，

直接运行  bash run.sh即可实现socket通信
