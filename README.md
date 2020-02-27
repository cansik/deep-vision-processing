# Deep Vision Processing
Deep computer-vision algorithms for [Processing](https://processing.org/).

### Idea
The idea behind this library is to provide a simple way to use (inference) neural networks for computer vision tasks inside Processing.

First of all a simple proof of concept will be created which implements the [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) to define where to store the weight and model data and to specify the final API.

#### API Structure
The API should support the following (✨ == `implemented`):

- Support for Model fetching
    - Local sketch installation
    - Global library installation
- YOLO
    - YOLOv3-tiny ✨
    - YOLOv3-spp ([spatial pyramid pooling](https://stackoverflow.com/a/55014630/1138326))
    - YOLOv3-320
    - YOLOv3-416
    - YOLOv3-608 ✨
    - YOLO 9K
- openVINO
    - Lightweight OpenPose
    - Face Detection

### About
Maintained by [cansik](https://github.com/cansik) with the help of the following dependencies:

- [bytedeco/javacv](https://github.com/bytedeco/javacv)
- [atduskgreg/opencv-processing](https://github.com/atduskgreg/opencv-processing)
