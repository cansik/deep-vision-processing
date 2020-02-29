package ch.bildspur.vision.deps;

import java.nio.file.Path;
import java.nio.file.Paths;

public class Repository {
    public static String repositoryRootUrl = "https://github.com/cansik/deep-vision-processing/releases/download/repository/";
    public static Path localStorageDirectory = Paths.get("networks");

    // names
    public static final Dependency COCONames = new Dependency("coco.names");
    public static final Dependency VOCNames = new Dependency("voc.names");
    public static final Dependency OpenImagesNames = new Dependency("openimages.names");
    public static final Dependency NineKNames = new Dependency("9k.names");

    // networks

    // object detection
    public static final Dependency YOLOv3Model = new Dependency("yolov3.cfg");
    public static final Dependency YOLOv3Weight = new Dependency("yolov3.weights");

    public static final Dependency YOLOv3TinyModel = new Dependency("yolov3-tiny.cfg");
    public static final Dependency YOLOv3TinyWeight = new Dependency("yolov3-tiny.weights");

    public static final Dependency YOLOv3SPPModel = new Dependency("yolov3-spp.cfg");
    public static final Dependency YOLOv3SPPWeight = new Dependency("yolov3-spp.weights");

    public static final Dependency YOLO9kModel = new Dependency("yolo9000.cfg");
    public static final Dependency YOLO9kWeight = new Dependency("yolo9000.weights");

    // pose estimation
    public static final Dependency LIPSingleHumanPoseEstimationModel = new Dependency("lip-single-human-pose-estimation.onnx");
    public static final Dependency MultiHumanPoseEstimationModel = new Dependency("multi-human-pose-estimation.onnx");
}
