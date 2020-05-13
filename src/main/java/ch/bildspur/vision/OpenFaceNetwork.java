package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.network.MultiProcessor;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import ch.bildspur.vision.result.VectorResult;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTorch;

public class OpenFaceNetwork extends BaseNeuralNetwork<VectorResult> implements MultiProcessingNetwork<VectorResult> {
    private final MultiProcessor<VectorResult> multiProcessor = new MultiProcessor<>(this);
    private Net net;
    private Path modelPath;

    public OpenFaceNetwork(Path modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    public boolean setup() {
        net = readNetFromTorch(modelPath.toAbsolutePath().toString());
        return !net.empty();
    }

    @Override
    public VectorResult run(Mat frame) {
        Mat inputBlob = blobFromImage(frame,
                1 / 255.0,
                new Size(96, 96),
                new Scalar(0.0),
                true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        Mat out = net.forward();

        // extract embeddings
        FloatPointer data = new FloatPointer(out.row(0).data());
        float[] vector = new float[out.size(1)];
        for(int i = 0; i < vector.length; i++) {
            vector[i] = data.get(i);
        }

        return new VectorResult(vector);
    }

    @Override
    public ResultList<VectorResult> runByDetections(PImage image, ResultList<ObjectDetectionResult> detections) {
        return multiProcessor.runByDetections(image, detections);
    }

    @Override
    public ResultList<VectorResult> runByDetections(Mat frame, ResultList<ObjectDetectionResult> detections) {
        return multiProcessor.runByDetections(frame, detections);
    }
}
