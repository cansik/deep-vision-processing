package ch.bildspur.vision.network;

import ch.bildspur.vision.result.NetworkResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PImage;

import java.util.List;

import static ch.bildspur.vision.util.CvProcessingUtils.createValidROI;

public class MultiProcessor<R extends NetworkResult> implements MultiProcessingNetwork<R> {
    BaseNeuralNetwork<R> network;

    public MultiProcessor(BaseNeuralNetwork<R> network) {
        this.network = network;
    }

    @Override
    public ResultList<R> runByDetections(PImage image, List<ObjectDetectionResult> detections) {
        Mat frame = network.convertToMat(image);
        return runByDetections(frame, detections);
    }

    @Override
    public ResultList<R> runByDetections(Mat frame, List<ObjectDetectionResult> detections) {
        ResultList<R> results = new ResultList<>();

        for (ObjectDetectionResult detection : detections) {
            Mat roi = new Mat(frame, createValidROI(frame.size(), detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight()));
            R result = network.run(roi);
            results.add(result);
            roi.release();
        }

        return results;
    }

    @Override
    public boolean setup() {
        return network.setup();
    }

    @Override
    public R run(Mat frame) {
        return network.run(frame);
    }
}
