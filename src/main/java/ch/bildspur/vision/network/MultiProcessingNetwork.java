package ch.bildspur.vision.network;

import ch.bildspur.vision.result.NetworkResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PImage;

public interface MultiProcessingNetwork<R extends NetworkResult> extends NeuralNetwork<R> {
    ResultList<R> runByDetections(PImage image, ResultList<ObjectDetectionResult> detections);

    ResultList<R> runByDetections(Mat frame, ResultList<ObjectDetectionResult> detections);
}
