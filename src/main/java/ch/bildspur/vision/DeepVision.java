package ch.bildspur.vision;

import ch.bildspur.vision.deps.Dependency;
import ch.bildspur.vision.deps.Repository;
import processing.core.PApplet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DeepVision {

    public DeepVision(PApplet sketch) {
        Repository.localStorageDirectory = Paths.get(sketch.sketchPath("networks"));
    }

    private void prepareDependencies(Dependency... dependencies) {
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

    // yolo

    private YOLONetwork createYOLONetwork(Dependency model, Dependency weights, Dependency names, int size) {
        prepareDependencies(model, weights, names);

        YOLONetwork network = new YOLONetwork(
                model.getPath(),
                weights.getPath(),
                size, size
        );

        network.loadNames(names.getPath());
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

    public MultiHumanPoseNetwork createMultiHumanPoseEstimation() {
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
}
