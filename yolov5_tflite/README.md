# YOLOv5 PyTorch to TFLite example

Export [ultralytics/YOLOv5](https://github.com/ultralytics/yolov5) to TFLite and evaluate on COCO val dataset.

## URLs

- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/yolov5/issues/251
- https://github.com/ultralytics/yolov5/issues/5481

## Procedure

```
# clone yolov5 source
# tested with c43439aa31afdca9d1adbd1cc35b57bfb95b442d
git clone git@github.com:ultralytics/yolov5.git
cd yolov5

# docker build and run
t=ultralytics/yolov5:latest && \
docker pull $t && \
docker run -it --ipc=host --gpus all -v "$(pwd)":/usr/src/app -v "$(pwd)"/datasets:/usr/src/datasets $t

# install tensorflow for TFLite export
pip install "tensorflow>=2.4.1"  # "2.4.1" from yolov5/requirements.txt

# evaluate YOLOv5s pytorch model (baseline)
python val.py --weights yolov5s.pt --data data/coco.yaml

# export YOLOv5s TFLite fp16 model and evaluate
python export.py --weights yolov5s.pt --include tflite
python val.py --weights yolov5s-fp16.tflite --data data/coco.yaml

# export YOLOv5s TFLite int8 model and evaluate
python export.py --weights yolov5s.pt --include tflite --int8
python val.py --weights yolov5s-int8.tflite --data data/coco.yaml
```

## Results

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |mAP<sup>val<br>0.75 |mAP<sup>val<br>Small |mAP<sup>val<br>Medium |mAP<sup>val<br>Large
|---                   |--- |---  |---  |---  |---  |---  |---
|YOLOv5s [PyTorch]     |640 |37.1 |56.3 |40.1 |21.9 |42.6 |47.2
|YOLOv5s [TFLite fp16] |640 |37.0 |56.1 |39.9 |21.5 |42.4 |47.5
|YOLOv5s [TFLite int8] |640 |31.4 |51.9 |33.0 |14.2 |36.3 |43.5

## Appendix: demo on edge devices (e.g., RPi)

```
# install tflite_runtime
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

git clone git@github.com:ultralytics/yolov5.git
cd yolov5

# run webcam demo
# note that .tflite weight file is the one prepared above
python detect.py --weight yolov5s-int8.tflite --source 0
```
