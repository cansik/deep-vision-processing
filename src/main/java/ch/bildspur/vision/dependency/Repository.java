package ch.bildspur.vision.dependency;

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
    public static final Dependency COCOLabels2014To2017 = new Dependency("coco-labels-2014_2017.txt");
    public static final Dependency COCOLabelsPaper = new Dependency("coco-labels-paper.txt");

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

    public static final Dependency SSDMobileNetV2COCOConfig = new Dependency("ssd_mobilenet_v2_coco_2018_03_29.pbtxt");
    public static final Dependency SSDMobileNetV2COCOWeight = new Dependency("ssd_mobilenet_v2_coco_2018_03_29.pb");

    // hand detection
    public static final Dependency HandTrackJSWeight = new Dependency("handtrackingjs.pb");
    public static final Dependency HandTrackJSConfig = new Dependency("handtrackingjs.pbtxt");

    public static final Dependency CMUHandYOLOv3Weights = new Dependency("cmu_hand.weights");
    public static final Dependency CMUHandYOLOv3Model = new Dependency("cmu_hand.cfg");

    public static final Dependency HaarCascadeHand = new Dependency("haarcascade_hand.xml");

    // text detection
    public static final Dependency TextBoxesProtoText = new Dependency("TextBoxes_icdar13.prototxt");
    public static final Dependency TextBoxesModel = new Dependency("TextBoxes_icdar13.caffemodel");

    public static final Dependency TextBoxesPlusPlusProtoText = new Dependency("TextBoxes_plusplus_icdar15.prototxt");
    public static final Dependency TextBoxesPlusPlusModel = new Dependency("TextBoxes_plusplus_icdar15.caffemodel");

    // text recognition
    public static final Dependency CRNNModel = new Dependency("model_crnn.t7");

    // super resolution
    public static final Dependency FSRCNNModel = new Dependency("FSRCNN_x2.pb");

    // masking
    public static final Dependency MaskRCNNInceptionv2Config = new Dependency("mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
    public static final Dependency MaskRCNNInceptionv2Weight = new Dependency("mask_rcnn_inception_v2_coco_2018_01_28.pb");

    // pose estimation
    public static final Dependency SingleHumanPoseEstimationModel = new Dependency("single-human-pose-estimation-l4.onnx");
    public static final Dependency MultiHumanPoseEstimationModel = new Dependency("multi-human-pose-estimation.onnx");

    // face recognition
    public static final Dependency ULFGFaceDetectorRFB320Simplified = new Dependency("version-RFB-320_simplified.onnx");
    public static final Dependency ULFGFaceDetectorSlim320Simplified = new Dependency("version-slim-320_simplified.onnx");
    public static final Dependency ULFGFaceDetectorRFB640Simplified = new Dependency("version-RFB-640_simplified.onnx");
    public static final Dependency ULFGFaceDetectorSlim640Simplified = new Dependency("version-slim-640_simplified.onnx");

    // facial landmark
    public static final Dependency FaceMarkLBFModel = new Dependency("lbfmodel.yaml");

    // classification
    public static final Dependency MNISTModel = new Dependency("mnist.onnx");

    public static final Dependency FERPlusEmotionModel = new Dependency("emotion-ferplus-8.onnx");

    public static final Dependency AgeNetProtoText = new Dependency("deploy_age.prototxt");
    public static final Dependency AgeNetModel = new Dependency("age_net.caffemodel");

    public static final Dependency GenderNetProtoText = new Dependency("deploy_gender.prototxt");
    public static final Dependency GenderNetModel = new Dependency("gender_net.caffemodel");

    // haar cascade
    public static final Dependency HaarCascadeFrontalFaceAlt = new Dependency("haarcascade_frontalface_alt.xml");

    // text models
    public static final Dependency TesseractEngBest = new Dependency("eng.traineddata");
}
