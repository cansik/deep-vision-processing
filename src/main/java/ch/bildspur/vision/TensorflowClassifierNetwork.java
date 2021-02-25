package ch.bildspur.vision;

import ch.bildspur.vision.network.ClassificationNetwork;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;

public class TensorflowClassifierNetwork extends ClassificationNetwork {
    private Path modelPath;

    public TensorflowClassifierNetwork(Path modelPath, int width, int height, String ... labels) {
        super(width, height, false, 1 / 255.0f, Scalar.all(0.0),
                false, true, 100.0f, labels);
        this.modelPath = modelPath;
    }

    @Override
    public Net createNetwork() {
        return readNetFromTensorflow(modelPath.toAbsolutePath().toString());
    }
}
