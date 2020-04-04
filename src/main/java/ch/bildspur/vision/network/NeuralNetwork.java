package ch.bildspur.vision.network;

import ch.bildspur.vision.result.NetworkResult;
import org.bytedeco.opencv.opencv_core.Mat;

public interface NeuralNetwork<R extends NetworkResult> {
    boolean setup();

    R run(Mat frame);
}
