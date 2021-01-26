package ch.bildspur.vision;

import ch.bildspur.vision.dependency.Dependency;
import ch.bildspur.vision.dependency.Repository;
import ch.bildspur.vision.util.ProcessingUtils;
import processing.core.PApplet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DeepVision {
    private PApplet sketch;
    private boolean storeNetworksInSketch = false;

    public DeepVision(PApplet sketch) {
        this.sketch = sketch;
    }

    public void storeNetworksGlobal() {
        storeNetworksInSketch = false;
    }

    public void storeNetworksInSketch() {
        storeNetworksInSketch = true;
    }

    public void clearRepository() {
        updateRepositoryPath();

        try {
            Files.list(Repository.localStorageDirectory)
                    .filter(e -> !Files.isDirectory(e))
                    .forEach(e -> {
                        try {
                            Files.delete(e);
                        } catch (IOException ex) {
                            ex.printStackTrace();
                        }
                    });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String getNetworkStoragePath() {
        return Repository.localStorageDirectory.toAbsolutePath().toString();
    }

    protected void updateRepositoryPath() {
        // decide where to store
        if (storeNetworksInSketch) {
            Repository.localStorageDirectory = Paths.get(sketch.sketchPath("networks"));
        } else {
            Repository.localStorageDirectory = Paths.get(ProcessingUtils.getLibPath(this), "networks");
        }
    }

    protected void prepareDependencies(Dependency... dependencies) {
        updateRepositoryPath();

        // download
        try {
            Files.createDirectories(Repository.localStorageDirectory);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (Dependency dependency : dependencies) {
            dependency.resolve();
        }
    }

    // cascade

    public CascadeClassifierNetwork createCascadeFrontalFace() {
        prepareDependencies(Repository.HaarCascadeFrontalFaceAlt);

        return new CascadeClassifierNetwork(Repository.HaarCascadeFrontalFaceAlt.getPath(), "face");
    }

    // yolo

    protected YOLONetwork createYOLONetwork(Dependency model, Dependency weights, Dependency names, int size) {
        prepareDependencies(model, weights, names);

        YOLONetwork network = new YOLONetwork(
                model.getPath(),
                weights.getPath(),
                size, size
        );

        network.loadLabels(names.getPath());
        return network;
    }

    public YOLONetwork createYOLOv3() {
        return createYOLOv3(608);
    }

    public YOLONetwork createYOLOv3(int inputSize) {
        return createYOLONetwork(Repository.YOLOv3Model, Repository.YOLOv3Weight, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv3Tiny() {
        return createYOLOv3Tiny(416);
    }

    public YOLONetwork createYOLOv3Tiny(int inputSize) {
        return createYOLONetwork(Repository.YOLOv3TinyModel, Repository.YOLOv3TinyWeight, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv3TinyPRN() {
        return createYOLOv3TinyPRN(416);
    }

    public YOLONetwork createYOLOv3TinyPRN(int inputSize) {
        return createYOLONetwork(Repository.YOLOv3TinyModelPRN, Repository.YOLOv3TinyWeightPRN, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv3OpenImages() {
        return createYOLOv3OpenImages(608);
    }

    public YOLONetwork createYOLOv3OpenImages(int inputSize) {
        return createYOLONetwork(Repository.YOLOv3OpenImagesModel, Repository.YOLOv3OpenImagesWeight, Repository.OpenImagesNames, inputSize);
    }

    public YOLONetwork createYOLOv3EfficientNet() {
        return createYOLOv3EfficientNet(416);
    }

    public YOLONetwork createYOLOv3EfficientNet(int inputSize) {
        return createYOLONetwork(Repository.EfficientNetB0Yolov3Model, Repository.EfficientNetB0Yolov3Weight, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv3SPP() {
        return createYOLOv3SPP(608);
    }

    public YOLONetwork createYOLOv3SPP(int inputSize) {
        return createYOLONetwork(Repository.YOLOv3SPPModel, Repository.YOLOv3SPPWeight, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv4() {
        return createYOLOv4(608);
    }

    public YOLONetwork createYOLOv4(int inputSize) {
        return createYOLONetwork(Repository.YOLOv4Model, Repository.YOLOv4Weight, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv4Tiny() {
        return createYOLOv4Tiny(416);
    }

    public YOLONetwork createYOLOv4Tiny(int inputSize) {
        return createYOLONetwork(Repository.YOLOv4TinyModel, Repository.YOLOv4TinyWeight, Repository.COCONames, inputSize);
    }

    public YOLONetwork createYOLOv3HandDetector() {
        return createYOLOv3HandDetector(416);
    }

    public YOLONetwork createYOLOv3HandDetector(int inputSize) {
        prepareDependencies(Repository.CMUHandYOLOv3Model, Repository.CMUHandYOLOv3Weights);

        YOLONetwork network = new YOLONetwork(
                Repository.CMUHandYOLOv3Model.getPath(), Repository.CMUHandYOLOv3Weights.getPath(),
                inputSize, inputSize
        );
        network.setLabels("hand");
        return network;
    }

    public YOLONetwork createCrossHandDetector() {
        return createYOLOv3HandDetector(416);
    }

    public YOLONetwork createCrossHandDetector(int inputSize) {
        prepareDependencies(Repository.CrossHandsYOLOv3Model, Repository.CrossHandsYOLOv3Weights);

        YOLONetwork network = new YOLONetwork(
                Repository.CrossHandsYOLOv3Model.getPath(), Repository.CrossHandsYOLOv3Weights.getPath(),
                inputSize, inputSize
        );
        network.setLabels("hand");
        return network;
    }

    public YOLONetwork createCrossHandDetectorTinyPRN() {
        return createCrossHandDetectorTinyPRN(416);
    }

    public YOLONetwork createCrossHandDetectorTinyPRN(int inputSize) {
        prepareDependencies(Repository.CrossHandsYOLOv3TinyPRNModel, Repository.CrossHandsYOLOv3TinyPRNWeights);

        YOLONetwork network = new YOLONetwork(
                Repository.CrossHandsYOLOv3TinyPRNModel.getPath(), Repository.CrossHandsYOLOv3TinyPRNWeights.getPath(),
                inputSize, inputSize
        );
        network.setLabels("hand");
        return network;
    }

    public YOLONetwork createYOLOv3TinyHandDetector() {
        return createYOLOv3TinyHandDetector(416);
    }

    public YOLONetwork createYOLOv3TinyHandDetector(int inputSize) {
        prepareDependencies(Repository.CMUHandYOLOv3TinyModel, Repository.CMUHandYOLOv3TinyWeights);

        YOLONetwork network = new YOLONetwork(
                Repository.CMUHandYOLOv3TinyModel.getPath(), Repository.CMUHandYOLOv3TinyWeights.getPath(),
                inputSize, inputSize
        );
        network.setLabels("hand");
        return network;
    }

    public MaskRCNN createMaskRCNN() {
        prepareDependencies(Repository.MaskRCNNInceptionv2Config, Repository.MaskRCNNInceptionv2Weight, Repository.COCOLabelsPaper);
        return new MaskRCNN(Repository.MaskRCNNInceptionv2Config.getPath(), Repository.MaskRCNNInceptionv2Weight.getPath(), Repository.COCOLabelsPaper.getPath());
    }

    // pose

    public SingleHumanPoseNetwork createSingleHumanPoseEstimation() {
        prepareDependencies(Repository.SingleHumanPoseEstimationModel);
        return new SingleHumanPoseNetwork(Repository.SingleHumanPoseEstimationModel.getPath());
    }

    // face detection
    public ULFGFaceDetectionNetwork createULFGFaceDetectorRFB320() {
        prepareDependencies(Repository.ULFGFaceDetectorRFB320Simplified);
        return new ULFGFaceDetectionNetwork(Repository.ULFGFaceDetectorRFB320Simplified.getPath(), 320, 240);
    }

    public ULFGFaceDetectionNetwork createULFGFaceDetectorSlim320() {
        prepareDependencies(Repository.ULFGFaceDetectorSlim320Simplified);
        return new ULFGFaceDetectionNetwork(Repository.ULFGFaceDetectorSlim320Simplified.getPath(), 320, 240);
    }

    public ULFGFaceDetectionNetwork createULFGFaceDetectorRFB640() {
        prepareDependencies(Repository.ULFGFaceDetectorRFB640Simplified);
        return new ULFGFaceDetectionNetwork(Repository.ULFGFaceDetectorRFB640Simplified.getPath(), 640, 480);
    }

    public ULFGFaceDetectionNetwork createULFGFaceDetectorSlim640() {
        prepareDependencies(Repository.ULFGFaceDetectorSlim640Simplified);
        return new ULFGFaceDetectionNetwork(Repository.ULFGFaceDetectorSlim640Simplified.getPath(), 640, 480);
    }

    // facial landmark
    public FacemarkLBFNetwork createFacemarkLBF() {
        prepareDependencies(Repository.FaceMarkLBFModel);
        return new FacemarkLBFNetwork(Repository.FaceMarkLBFModel.getPath());
    }

    // classification
    public MNISTNetwork createMNISTClassifier() {
        prepareDependencies(Repository.MNISTModel);
        return new MNISTNetwork(Repository.MNISTModel.getPath());
    }

    public FERPlusEmotionNetwork createFERPlusEmotionClassifier() {
        prepareDependencies(Repository.FERPlusEmotionModel);
        return new FERPlusEmotionNetwork(Repository.FERPlusEmotionModel.getPath());
    }

    public AgeNetwork createAgeClassifier() {
        prepareDependencies(Repository.AgeNetProtoText, Repository.AgeNetModel);
        return new AgeNetwork(Repository.AgeNetProtoText.getPath(), Repository.AgeNetModel.getPath());
    }

    public GenderNetwork createGenderClassifier() {
        prepareDependencies(Repository.GenderNetProtoText, Repository.GenderNetModel);
        return new GenderNetwork(Repository.GenderNetProtoText.getPath(), Repository.GenderNetModel.getPath());
    }

    // ssd mobilenet

    public SSDMobileNetwork createHandDetector() {
        return createHandDetector(300);
    }

    public SSDMobileNetwork createHandDetector(int inputSize) {
        prepareDependencies(Repository.HandTrackJSWeight, Repository.HandTrackJSConfig);
        return new SSDMobileNetwork(Repository.HandTrackJSWeight.getPath(), Repository.HandTrackJSConfig.getPath(),
                inputSize, inputSize, 0.5f, "hand");
    }

    public SSDMobileNetwork createMobileNetV2() {
        return createMobileNetV2(300);
    }

    public SSDMobileNetwork createMobileNetV2(int inputSize) {
        prepareDependencies(
                Repository.SSDMobileNetV2COCOWeight,
                Repository.SSDMobileNetV2COCOConfig,
                Repository.COCOLabelsPaper);

        SSDMobileNetwork network = new SSDMobileNetwork(Repository.SSDMobileNetV2COCOWeight.getPath(), Repository.SSDMobileNetV2COCOConfig.getPath(),
                inputSize, inputSize, 0.5f);
        network.loadLabels(Repository.COCOLabelsPaper.getPath());
        return network;
    }

    public TextBoxesNetwork createTextBoxesDetector() {
        prepareDependencies(Repository.TextBoxesProtoText, Repository.TextBoxesModel);
        return new TextBoxesNetwork(Repository.TextBoxesProtoText.getPath(), Repository.TextBoxesModel.getPath());
    }

    public TesseractNetwork createTesseractRecognizer() {
        prepareDependencies(Repository.TesseractEngBest);
        return new TesseractNetwork(Repository.TesseractEngBest.getPath(), "eng");
    }

    // super resolution

    public SuperResolutionNetwork createSuperResolutionNetwork(String name, int scale, Dependency model) {
        prepareDependencies(model);
        return new SuperResolutionNetwork(model.getPath(), name, scale);
    }

    public SuperResolutionNetwork createSuperResolutionFSCRNN(int scale) {
        String name = "fsrcnn";
        switch (scale) {
            case 3:
                return createSuperResolutionNetwork(name, scale, Repository.FSRCNNx3Model);
            case 4:
                return createSuperResolutionNetwork(name, scale, Repository.FSRCNNx4Model);
            default:
                return createSuperResolutionNetwork(name, 2, Repository.FSRCNNx2Model);
        }
    }

    public SuperResolutionNetwork createSuperResolutionLapSRN(int scale) {
        String name = "lapsrn";
        switch (scale) {
            case 4:
                return createSuperResolutionNetwork(name, scale, Repository.LapSRNx4Model);
            case 8:
                return createSuperResolutionNetwork(name, scale, Repository.LapSRNx8Model);
            default:
                return createSuperResolutionNetwork(name, 2, Repository.LapSRNx2Model);
        }
    }

    public SuperResolutionNetwork createSuperResolutionESPCN(int scale) {
        String name = "espcn";
        switch (scale) {
            case 3:
                return createSuperResolutionNetwork(name, scale, Repository.ESPCNx3Model);
            case 4:
                return createSuperResolutionNetwork(name, scale, Repository.ESPCNx4Model);
            default:
                return createSuperResolutionNetwork(name, 2, Repository.ESPCNx2Model);
        }
    }

    public SuperResolutionNetwork createSuperResolutionEDSR(int scale) {
        String name = "edsr";
        switch (scale) {
            case 3:
                return createSuperResolutionNetwork(name, scale, Repository.EDSRx3Model);
            case 4:
                return createSuperResolutionNetwork(name, scale, Repository.EDSRx4Model);
            default:
                return createSuperResolutionNetwork(name, 2, Repository.EDSRx2Model);
        }
    }

    // style transfer
    public StyleTransferNetwork createStyleTransfer(Dependency model) {
        prepareDependencies(model);
        return new StyleTransferNetwork(model.getPath());
    }

    public StyleTransferNetwork createStyleTransfer() {
        return createStyleTransfer(Repository.ECCV16CompositionVII);
    }

    // depth estimation

    public MidasNetwork createMidasNetwork() {
        prepareDependencies(Repository.MidasNetModel);
        return new MidasNetwork(Repository.MidasNetModel.getPath());
    }
}
