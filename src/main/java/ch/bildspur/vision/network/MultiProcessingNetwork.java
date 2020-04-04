package ch.bildspur.vision.network;

import ch.bildspur.vision.result.NetworkResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PImage;

import java.util.List;

public interface MultiProcessingNetwork<R extends NetworkResult> extends NeuralNetwork<R> {
    List<R> runByDetections(PImage image, List<ObjectDetectionResult> detections);

    List<R> runByDetections(Mat frame, List<ObjectDetectionResult> detections);
}
