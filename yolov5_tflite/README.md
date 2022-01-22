# YOLOv5 PyTorch to TFLite example

## URLs

- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/yolov5/issues/251
- https://github.com/ultralytics/yolov5/issues/5481

## Procedure

```
PROJ_DIR=$HOME
cd $PROJ_DIR
git clone git@github.com:ultralytics/yolov5.git
cd yolov5

# docker build and run
t=ultralytics/yolov5:latest && \
docker pull $t && \
docker run -it --ipc=host --gpus all -v "$(pwd)":/usr/src/app -v "$(pwd)"/datasets:/usr/src/datasets $t

# install tensorflow for TFLite export
pip install "tensorflow>=2.4.1"  # "2.4.1" from yolov5/requirements.txt

# evaluate YOLOv5s pytorch model (baseline)
python val.py --weights yolov5s.pt

# export YOLOv5s TFLite fp16 model and evaluate
python export.py --weights yolov5s.pt --include tflite
python val.py --weights yolov5s-fp16.tflite

# export YOLOv5s TFLite int8 model and evaluate
python export.py --weights yolov5s.pt --include tflite --int8
python val.py --weights yolov5s-int8.tflite
```
