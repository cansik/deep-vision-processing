package ch.bildspur.vision;

import ch.bildspur.vision.deps.Dependency;
import ch.bildspur.vision.deps.Repository;
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

    private void prepareDependencies(Dependency... dependencies) {
        // decide where to store
        if (storeNetworksInSketch) {
            Repository.localStorageDirectory = Paths.get(sketch.sketchPath("networks"));
        } else {
            Repository.localStorageDirectory = Paths.get(ProcessingUtils.getLibPath(this), "networks");
        }

        System.out.println("Storing networks to: " + Repository.localStorageDirectory.toAbsolutePath().toString());

        // download
        try {
            Files.createDirectories(Repository.localStorageDirectory);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (Dependency dependency : dependencies) {
            System.out.println("preparing " + dependency.getName() + "...");
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
        return createYOLONetwork(Repository.YOLOv3Model, Repository.YOLOv3Weight, Repository.COCONames, 608);
    }

    public YOLONetwork createYOLOv3Tiny() {
        return createYOLONetwork(Repository.YOLOv3TinyModel, Repository.YOLOv3TinyWeight, Repository.COCONames, 416);
    }

    public YOLONetwork createYOLOv3SPP() {
        return createYOLONetwork(Repository.YOLOv3SPPModel, Repository.YOLOv3SPPWeight, Repository.COCONames, 608);
    }

    // pose

    public SingleHumanPoseNetwork createSingleHumanPoseEstimation() {
        prepareDependencies(Repository.SingleHumanPoseEstimationModel);
        return new SingleHumanPoseNetwork(Repository.SingleHumanPoseEstimationModel.getPath());
    }

    public MultiHumanPoseNetwork _createMultiHumanPoseEstimation() {
        prepareDependencies(Repository.MultiHumanPoseEstimationModel);
        return new MultiHumanPoseNetwork(Repository.MultiHumanPoseEstimationModel.getPath());
    }

    // face recognition
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

    // hand recognition

    public HandDetectionNetwork _createHandDetector() {
        prepareDependencies(Repository.HandTrackJSWeight, Repository.HandTrackJSConfig);
        return new HandDetectionNetwork(Repository.HandTrackJSWeight.getPath(), Repository.HandTrackJSConfig.getPath());
    }
}
