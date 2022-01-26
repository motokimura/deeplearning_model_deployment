# YOLOv3 PyTorch to OpenVINO example

Export [mmdetection YOLOv3](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/README.md)
to OpenVINO and evaluate on COCO val dataset.

## URLs

- https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/README.md
- https://github.com/open-mmlab/mmdeploy

## Procedure

```
git clone git@github.com:open-mmlab/mmdeploy.git  # tested with 0556feec79ffbb8f8c371343175c0b8d55b09841
cd mmdeploy

docker build docker/CPU/ -t mmdeploy:master-cpu
docker run -it -p 8080:8081 -v $HOME/data:/root/workspace/mmdeploy/data mmdeploy:master-cpu

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

2002 sec to infer 5,000 images (Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz).

### OpenVINO

```
cd /root/workspace/mmdeploy

python tools/deploy.py \
    configs/mmdet/detection/detection_openvino_dynamic-800x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    $PATH_TO_MMDET/demo/demo.jpg \
    --work-dir work_openvino_dynamic

python tools/test.py \
    configs/mmdet/detection/detection_openvino_dynamic-800x1344.py \
    $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    --model work_openvino_dynamic/end2end.xml \
    --metrics bbox
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

1152 sec to infer 5,000 images (Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz).
