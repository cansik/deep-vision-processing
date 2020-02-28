package ch.bildspur.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public abstract class PoseNetwork<R> extends DeepNeuralNetwork<R> {
    private Path modelPath;
    private Net net;

    private final int inputHeight;
    private final int inputWidth;

    public PoseNetwork(Path modelPath, int inputWidth, int inputHeight) {
        this.modelPath = modelPath;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
    }

    @Override
    public boolean setup() {
        net = createNetwork();

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    public abstract Net createNetwork();

    public Mat[] extractHeatMaps(Mat frame) {
        Size inputSize = new Size(inputWidth, inputHeight);
        Scalar zeroScalar = new Scalar(0.0, 0.0, 0.0, 0.0);
        Mat inputBlob = blobFromImage(frame, 1 / 255.0, inputSize, zeroScalar, true, false, CV_32F);

        // inferencing
        net.setInput(inputBlob);
        Mat output = net.forward();

        return splitNetOutputBlobToParts(output, frame.size());
    }

    private Mat[] splitNetOutputBlobToParts(Mat output, Size inputSize) {
        int nParts = output.size(1);
        int matHeight = output.size(2);
        int matWidth = output.size(3);

        Mat[] mats = new Mat[nParts];
        for (int i = 0; i < nParts; i++) {
            Mat m = new Mat(matHeight, matWidth, CV_32F, output.ptr(0, i));
            resize(m, m, inputSize);
            mats[i] = m;
        }

        return mats;
    }

    public Path getModelPath() {
        return modelPath;
    }

    public Net getNet() {
        return net;
    }
}
