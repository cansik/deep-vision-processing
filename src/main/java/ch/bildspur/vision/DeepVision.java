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

    private YOLONetwork createYOLONetwork(Dependency model, Dependency weights, Dependency names, int size) {
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

    public YOLONetwork createYOLOv3SPP() {
        return createYOLOv3SPP(608);
    }

    public YOLONetwork createYOLOv3SPP(int inputSize) {
        return createYOLONetwork(Repository.YOLOv3SPPModel, Repository.YOLOv3SPPWeight, Repository.COCONames, inputSize);
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

    public FSRCNNNetwork createFSCRNN() {
        prepareDependencies(Repository.FSRCNNModel);
        return new FSRCNNNetwork(Repository.FSRCNNModel.getPath());
    }
}
