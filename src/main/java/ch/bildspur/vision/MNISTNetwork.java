package ch.bildspur.vision;

import ch.bildspur.vision.network.ClassificationNetwork;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

public class MNISTNetwork extends ClassificationNetwork {
    private Path modelPath;

    public MNISTNetwork(Path modelPath) {
        super(28, 28, true, 1 / 255.0f, Scalar.all(0.0), false, true, 100.0f,
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
        this.modelPath = modelPath;
    }

    @Override
    public Net createNetwork() {
        return readNetFromONNX(modelPath.toAbsolutePath().toString());
    }
}
