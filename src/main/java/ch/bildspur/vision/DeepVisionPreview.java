package ch.bildspur.vision;

import ch.bildspur.vision.dependency.Repository;
import processing.core.PApplet;

public class DeepVisionPreview extends DeepVision {
    public DeepVisionPreview(PApplet sketch) {
        super(sketch);
    }

    public MultiHumanPoseNetwork createMultiHumanPoseEstimation() {
        prepareDependencies(Repository.MultiHumanPoseEstimationModel);
        return new MultiHumanPoseNetwork(Repository.MultiHumanPoseEstimationModel.getPath());
    }

    public TextBoxesNetwork createTextBoxesPlusPlusDetector() {
        prepareDependencies(Repository.TextBoxesPlusPlusProtoText, Repository.TextBoxesPlusPlusModel);
        return new TextBoxesNetwork(Repository.TextBoxesPlusPlusProtoText.getPath(), Repository.TextBoxesPlusPlusModel.getPath());
    }

    public CRNNNetwork createCRNN() {
        prepareDependencies(Repository.CRNNModel);
        return new CRNNNetwork(Repository.CRNNModel.getPath());
    }

    public CascadeClassifierNetwork createCascadeHand() {
        prepareDependencies(Repository.HaarCascadeHand);
        return new CascadeClassifierNetwork(Repository.HaarCascadeHand.getPath(), "hand");
    }

    public YOLONetwork createYOLOv4() {
        return createYOLOv4(608);
    }

    public YOLONetwork createYOLOv4(int inputSize) {
        return createYOLONetwork(Repository.YOLOv4Model, Repository.YOLOv4Weight, Repository.COCONames, inputSize);
    }
}
