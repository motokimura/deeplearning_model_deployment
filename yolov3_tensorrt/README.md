# YOLOv3 PyTorch to TensorRT example

Export [mmdetection YOLOv3](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/README.md)
to TensorRT and evaluate on COCO val dataset.

## URLs

- https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/README.md
- https://github.com/open-mmlab/mmdeploy

## Procedure

```
git clone git@github.com:open-mmlab/mmdeploy.git  # tested with 57a9d9b6425d7890084cc73ef948cf3a42e9adc4
cd mmdeploy

docker build docker/GPU/ -t mmdeploy:master-gpu --build-arg CUDA=11.3 --build-arg TORCH_VERSION=1.10.0 --build-arg TORCHVISION_VERSION=0.11.1
docker run --gpus all -it -p 8080:8081 -v $HOME/data:/root/workspace/mmdeploy/data mmdeploy:master-gpu

pip install mmdet  # tested with 2.20.0

cd /root/workspace
git clone https://github.com/open-mmlab/mmdetection.git

cd mmdetection
mkdir -p checkpoints/yolo
cd checkpoints/yolo
wget https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth

PATH_TO_MMDET=/root/workspace/mmdetection
```

### PyTorch (baseline)

```
cd $PATH_TO_MMDET

ln -s /root/workspace/mmdeploy/data data

python tools/test.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    --eval bbox
```

output:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.566
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.353
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.594
```

85 sec to infer 5,000 images (RTX 3080, nvidia-driver 510.39.01).

### tensorrt_dynamic

```
cd /root/workspace/mmdeploy

python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_tensorrt_dynamic \
    --device cuda:0

python tools/test.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    --model work_tensorrt_dynamic/end2end.engine \
    --device cuda:0 \
    --metrics bbox
```

output:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.593
```

73 sec to infer 5,000 images (RTX 3080, nvidia-driver 510.39.01).

### tensorrt-fp16_dynamic

```
cd /root/workspace/mmdeploy

python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-fp16_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_tensorrt-fp16_dynamic \
    --device cuda:0

python tools/test.py \
    configs/mmdet/detection/detection_tensorrt-fp16_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    --model work_tensorrt-fp16_dynamic/end2end.engine \
    --device cuda:0 \
    --metrics bbox
```

output:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.592
```

43 sec to infer 5,000 images (RTX 3080, nvidia-driver 510.39.01).

### tensorrt-int8_dynamic

```
cd /root/workspace/mmdeploy

python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_tensorrt-int8_dynamic \
    --device cuda:0

python tools/test.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    --model work_tensorrt-int8_dynamic/end2end.engine \
    --device cuda:0 \
    --metrics bbox
```

output:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.593
```

43 sec to infer 5,000 images (RTX 3080, nvidia-driver 510.39.01).

The performance of `tensorrt-fp16_dynamic` and `tensorrt-int8_dynamic` models are the same.
This may be because TensorRT chooses fp16 kernel instead of int8 even in `tensorrt-int8_dynamic` mode.
See [this issue](https://github.com/open-mmlab/mmdeploy/issues/52).
