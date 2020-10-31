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

    public DORNDepthEstimationNetwork createDORNDepthEstimation() {
        prepareDependencies(Repository.DORNDepthEstimationDeployPrototext, Repository.DORNDepthEstimationModel);
        return new DORNDepthEstimationNetwork(Repository.DORNDepthEstimationDeployPrototext.getPath(), Repository.DORNDepthEstimationModel.getPath());
    }

    public OpenFaceNetwork createOpenFaceNetwork() {
        prepareDependencies(Repository.OpenFaceNN4Small2v1);
        return new OpenFaceNetwork(Repository.OpenFaceNN4Small2v1.getPath());
    }

    public Face3DDFAV2Network create3DDFAV2() {
        prepareDependencies(Repository.Face3DDFAV2Model);
        return new Face3DDFAV2Network(Repository.Face3DDFAV2Model.getPath());
    }

    public MidasNetwork createMidasNetwork() {
        prepareDependencies(Repository.MidasNetModel);
        return new MidasNetwork(Repository.MidasNetModel.getPath());
    }
}
