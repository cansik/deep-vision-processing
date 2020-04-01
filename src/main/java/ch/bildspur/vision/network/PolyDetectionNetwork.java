package ch.bildspur.vision.network;

import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PImage;

import java.util.List;

public interface PolyDetectionNetwork<T> {
    List<T> runByDetections(PImage image, List<ObjectDetectionResult> detections);

    List<T> runByDetections(Mat frame, List<ObjectDetectionResult> detections);
}
