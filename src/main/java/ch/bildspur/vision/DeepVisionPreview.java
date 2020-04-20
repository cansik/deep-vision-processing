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

    public StyleTransferNetwork createStyleTransfer() {
        prepareDependencies(Repository.ECCV16LaMuse);
        return new StyleTransferNetwork(Repository.ECCV16LaMuse.getPath());
    }
}
