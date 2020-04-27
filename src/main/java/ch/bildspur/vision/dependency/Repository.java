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
    public static final Dependency YOLOv4Model = new Dependency("yolov4.cfg");
    public static final Dependency YOLOv4Weight = new Dependency("yolov4.weights");

    public static final Dependency YOLOv3Model = new Dependency("yolov3.cfg");
    public static final Dependency YOLOv3Weight = new Dependency("yolov3.weights");

    public static final Dependency YOLOv3TinyModel = new Dependency("yolov3-tiny.cfg");
    public static final Dependency YOLOv3TinyWeight = new Dependency("yolov3-tiny.weights");

    public static final Dependency YOLOv3TinyModelPRN = new Dependency("yolov3-tiny-prn.cfg");
    public static final Dependency YOLOv3TinyWeightPRN = new Dependency("yolov3-tiny-prn.weights");

    public static final Dependency YOLOv3SPPModel = new Dependency("yolov3-spp.cfg");
    public static final Dependency YOLOv3SPPWeight = new Dependency("yolov3-spp.weights");

    public static final Dependency YOLO9kModel = new Dependency("yolo9000.cfg");
    public static final Dependency YOLO9kWeight = new Dependency("yolo9000.weights");

    public static final Dependency EfficientNetB0Yolov3Model = new Dependency("enet-coco.cfg");
    public static final Dependency EfficientNetB0Yolov3Weight = new Dependency("enetb0-coco_final.weights");

    public static final Dependency SSDMobileNetV2COCOConfig = new Dependency("ssd_mobilenet_v2_coco_2018_03_29.pbtxt");
    public static final Dependency SSDMobileNetV2COCOWeight = new Dependency("ssd_mobilenet_v2_coco_2018_03_29.pb");

    // hand detection
    public static final Dependency HandTrackJSWeight = new Dependency("handtrackingjs.pb");
    public static final Dependency HandTrackJSConfig = new Dependency("handtrackingjs.pbtxt");

    public static final Dependency CMUHandYOLOv3Weights = new Dependency("cmu_hand.weights");
    public static final Dependency CMUHandYOLOv3Model = new Dependency("cmu_hand.cfg");

    public static final Dependency CMUHandYOLOv3TinyWeights = new Dependency("cmu_hand_yolov3_tiny.weights");
    public static final Dependency CMUHandYOLOv3TinyModel = new Dependency("cmu_hand_yolov3_tiny.cfg");

    public static final Dependency CrossHandsYOLOv3Weights = new Dependency("cross-hands.weights");
    public static final Dependency CrossHandsYOLOv3Model = new Dependency("cross-hands.cfg");

    public static final Dependency HaarCascadeHand = new Dependency("haarcascade_hand.xml");

    // text detection
    public static final Dependency TextBoxesProtoText = new Dependency("TextBoxes_icdar13.prototxt");
    public static final Dependency TextBoxesModel = new Dependency("TextBoxes_icdar13.caffemodel");

    public static final Dependency TextBoxesPlusPlusProtoText = new Dependency("TextBoxes_plusplus_icdar15.prototxt");
    public static final Dependency TextBoxesPlusPlusModel = new Dependency("TextBoxes_plusplus_icdar15.caffemodel");

    // text recognition
    public static final Dependency CRNNModel = new Dependency("model_crnn.t7");

    // super resolution
    public static final Dependency FSRCNNx2Model = new Dependency("FSRCNN_x2.pb");
    public static final Dependency FSRCNNx3Model = new Dependency("FSRCNN_x3.pb");
    public static final Dependency FSRCNNx4Model = new Dependency("FSRCNN_x4.pb");

    public static final Dependency FSRCNNx2ModelSmall = new Dependency("FSRCNN-small_x2.pb");
    public static final Dependency FSRCNNx3ModelSmall = new Dependency("FSRCNN-small_x3.pb");
    public static final Dependency FSRCNNx4ModelSmall = new Dependency("FSRCNN-small_x4.pb");

    public static final Dependency LapSRNx2Model = new Dependency("LapSRN_x2.pb");
    public static final Dependency LapSRNx4Model = new Dependency("LapSRN_x4.pb");
    public static final Dependency LapSRNx8Model = new Dependency("LapSRN_x8.pb");

    public static final Dependency ESPCNx2Model = new Dependency("ESPCN_x2.pb");
    public static final Dependency ESPCNx3Model = new Dependency("ESPCN_x3.pb");
    public static final Dependency ESPCNx4Model = new Dependency("ESPCN_x4.pb");

    public static final Dependency EDSRx2Model = new Dependency("EDSR_x2.pb");
    public static final Dependency EDSRx3Model = new Dependency("EDSR_x3.pb");
    public static final Dependency EDSRx4Model = new Dependency("EDSR_x4.pb");

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

    // style transfer
    public static final Dependency ECCV16CompositionVII = new Dependency("composition_vii.t7");
    public static final Dependency ECCV16LaMuse = new Dependency("la_muse.t7");
    public static final Dependency ECCV16StarryNight = new Dependency("starry_night.t7");
    public static final Dependency ECCV16TheWave = new Dependency("the_wave.t7");

    public static final Dependency InstanceNormCandy = new Dependency("candy.t7");
    public static final Dependency InstanceNormFeathers = new Dependency("feathers.t7");
    public static final Dependency InstanceNormLaMuse = new Dependency("in_la_muse.t7");
    public static final Dependency InstanceNormMosaic = new Dependency("mosaic.t7");
    public static final Dependency InstanceNormTheScream = new Dependency("the_scream.t7");
    public static final Dependency InstanceNormUdnie = new Dependency("udnie.t7");
}
