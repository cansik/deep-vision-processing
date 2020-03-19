# Deep Vision Processing
Deep computer-vision algorithms for [Processing](https://processing.org/).

### Idea
The idea behind this library is to provide a simple way to use (inference) neural networks for computer vision tasks inside Processing. Mainly portability and easy-to-use are the primary goals of this library. Because of that (and javacv), no GPU support at the moment. 

First of all a simple proof of concept will be created which implements the [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) to define where to store the weight and model data and to specify the final API. (*This has already been done!*)

#### API Structure
The API should support the following (✨ = `implemented`):

- Support for model & weights fetching
    - Local sketch installation ✨
    - Global library installation (not yet supported)
- YOLO
    - YOLOv3-tiny ✨
    - YOLOv3-spp ([spatial pyramid pooling](https://stackoverflow.com/a/55014630/1138326)) ✨
    - YOLOv3 (608) ✨
    - ~~YOLO 9K~~ ([not supported by OpenCV](https://answers.opencv.org/question/180425/opencv-darknet-error-when-initializing-darknet/?answer=180441#post-id-180441))
- face detection
    - Ultra-Light-Fast-Generic-Face-Detector-1MB RFB (~30 FPS on CPU) ✨
    - Ultra-Light-Fast-Generic-Face-Detector-1MB Slim (~40 FPS on CPU) ✨
- openPose
    - Single Human Pose Detection based on lightweight openpose ✨
    - Multi Human Pose Detection (currently struggling with the partial affinity fields 🤷🏻‍♂️ help?)
 - classification
    - MNIST CNN ✨
    - FER+ Emotion ✨
    - Age Net ✨
    - Gender Net ✨
- openVINO (support is on it's way [javacpp-presets](https://github.com/bytedeco/javacpp-presets/pull/820))
    - Lightweight OpenPose (Multi-Person)
    - Face Detection
    - Facial Landmark Detection

### About
Maintained by [cansik](https://github.com/cansik) with the help of the following dependencies:

- [bytedeco/javacv](https://github.com/bytedeco/javacv)
- [atduskgreg/opencv-processing](https://github.com/atduskgreg/opencv-processing)

Stock images from the following peoples have been used:

- yoga.jpg by Yogendra Singh from Pexels
- office.jpg by [fauxels](https://www.pexels.com/@fauxels) from Pexels
- faces.png by [shvetsa](https://www.pexels.com/@shvetsa) from Pexels