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

    public MaskRCNN createMaskRCNN() {
        prepareDependencies(Repository.MaskRCNNInceptionv2Config, Repository.MaskRCNNInceptionv2Weight);
        return new MaskRCNN(Repository.MaskRCNNInceptionv2Config.getPath(), Repository.MaskRCNNInceptionv2Weight.getPath());
    }
}
