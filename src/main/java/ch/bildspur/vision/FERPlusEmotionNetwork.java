package ch.bildspur.vision;

import ch.bildspur.vision.network.ClassificationNetwork;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

public class FERPlusEmotionNetwork extends ClassificationNetwork {
    private Path modelPath;

    public FERPlusEmotionNetwork(Path modelPath) {
        super(64, 64, true, 1, Scalar.all(0.0), false, true,
                "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt");
        this.modelPath = modelPath;
    }

    @Override
    public Net createNetwork() {
        return readNetFromONNX(modelPath.toAbsolutePath().toString());
    }
}
