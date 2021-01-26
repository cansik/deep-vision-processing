package ch.bildspur.vision;

import ch.bildspur.vision.network.ClassificationNetwork;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;

public class AgeNetwork extends ClassificationNetwork {
    private Path protoTextPath;
    private Path modelPath;

    public AgeNetwork(Path protoTextPath, Path modelPath) {
        super(227, 227,
                false, 1,
                new Scalar(78.4263377603, 87.7689143744, 114.895847746, 0.0),
                false, false, 1.0f,
                "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100");

        this.protoTextPath = protoTextPath;
        this.modelPath = modelPath;
    }

    @Override
    public Net createNetwork() {
        return readNetFromCaffe(
                protoTextPath.toAbsolutePath().toString(),
                modelPath.toAbsolutePath().toString()
        );
    }
}
