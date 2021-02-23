package ch.bildspur.vision;

import ch.bildspur.vision.dependency.Repository;
import processing.core.PApplet;

public class DeepVisionPreview extends DeepVision {
    public DeepVisionPreview(PApplet sketch) {
        super(sketch);
    }

    public DeepVisionPreview(PApplet sketch, boolean enableCUDABackend) {
        super(sketch, enableCUDABackend);
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

    public DORNDepthEstimationNetwork createDORNDepthEstimation() {
        prepareDependencies(Repository.DORNDepthEstimationDeployPrototext, Repository.DORNDepthEstimationModel);
        return new DORNDepthEstimationNetwork(Repository.DORNDepthEstimationDeployPrototext.getPath(), Repository.DORNDepthEstimationModel.getPath());
    }

    public Face3DDFAV2Network create3DDFAV2() {
        prepareDependencies(Repository.Face3DDFAV2Model);
        return new Face3DDFAV2Network(Repository.Face3DDFAV2Model.getPath());
    }

    public MidasNetwork createMidasNetworkSmall() {
        prepareDependencies(Repository.MidasNet21SmallModel);
        return new MidasNetwork(Repository.MidasNet21SmallModel.getPath());
    }
}
