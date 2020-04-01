package ch.bildspur.vision.network;

import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PImage;

import java.util.ArrayList;
import java.util.List;

import static ch.bildspur.vision.util.CvProcessingUtils.createValidROI;

public class PolyDetector<T> implements PolyDetectionNetwork<T> {
    DeepNeuralNetwork<T> network;

    public PolyDetector(DeepNeuralNetwork<T> network) {
        this.network = network;
    }

    @Override
    public List<T> runByDetections(PImage image, List<ObjectDetectionResult> detections) {
        Mat frame = network.convertToMat(image);
        return runByDetections(frame, detections);
    }

    @Override
    public List<T> runByDetections(Mat frame, List<ObjectDetectionResult> detections) {
        List<T> results = new ArrayList<>();

        for (ObjectDetectionResult detection : detections) {
            Mat roi = new Mat(frame, createValidROI(frame.size(), detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight()));
            T result = network.run(roi);
            results.add(result);
            roi.release();
        }

        return results;
    }
}
