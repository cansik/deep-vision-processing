package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;

public class HandDetectionNetwork extends ObjectDetectionNetwork {

    private Path modelPath;
    private Path configPath;
    protected Net net;

    private int width = 320;
    private int height = 240;

    public HandDetectionNetwork(Path modelPath, Path configPath) {
        this.modelPath = modelPath;
        this.configPath = configPath;
        this.setLabels("hand");
    }

    @Override
    public boolean setup() {
        net = readNetFromTensorflow(
                modelPath.toAbsolutePath().toString());

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    public List<ObjectDetectionResult> run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1,
                new Size(width, height),
                Scalar.all(0),
                false, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        return new ArrayList<>();
    }
}
