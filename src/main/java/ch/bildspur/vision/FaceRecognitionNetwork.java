package ch.bildspur.vision;

import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.transpose;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

public class FaceRecognitionNetwork extends DeepNeuralNetwork<List<ObjectDetectionResult>> {
    private Path modelPath;
    protected Net net;

    private int width;
    private int height;

    private float confidenceThreshold = 0.5f;

    public FaceRecognitionNetwork(Path modelPath, int width, int height) {
        this.modelPath = modelPath;
        this.width = width;
        this.height = height;
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(modelPath.toAbsolutePath().toString());

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
                1 / 255.0,
                new Size(width, height),
                new Scalar(127, 127, 127, 0),
                false, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        // extract boxes and scores
        Mat boxesOut = outs.get(0);
        Mat scoresOut = outs.get(1);

        Mat boxes = new Mat();
        Mat scores = new Mat();

        transpose(boxesOut.reshape(1, 4), boxes);
        transpose(scoresOut.reshape(1, 2), scores);

        for (int i = 0; i < boxes.rows(); i++) {
            FloatPointer data = new FloatPointer(boxes.row(i).data());
            data.get(0);
        }

        return new ArrayList<>();
    }

}
