package ch.bildspur.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.StringVector;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public abstract class PoseNetwork<R> extends DeepNeuralNetwork<R> {
    private Path modelPath;
    protected Net net;

    private final int inputHeight;
    private final int inputWidth;

    protected double mean;
    protected Scalar meanScalar;
    protected double scale;

    public PoseNetwork(Path modelPath, int inputWidth, int inputHeight, double scale, double mean) {
        this(modelPath, inputWidth, inputHeight, scale, mean, new Scalar(mean, mean, mean, 0.0));
    }

    public PoseNetwork(Path modelPath, int inputWidth, int inputHeight, double scale, double mean, Scalar meanScalar) {
        this.modelPath = modelPath;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.scale = scale;
        this.mean = mean;
        this.meanScalar = meanScalar;
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

    protected Mat createInputBlob(Mat frame) {
        Size inputSize = new Size(inputWidth, inputHeight);
        return blobFromImage(frame, 1.0 / scale, inputSize, meanScalar, true, false, CV_32F);
    }

    protected Mat[] splitNetOutputBlobToParts(Mat output, Size inputSize, boolean resizeMat) {
        int nParts = output.size(1);
        int matHeight = output.size(2);
        int matWidth = output.size(3);

        Mat[] mats = new Mat[nParts];
        for (int i = 0; i < nParts; i++) {
            Mat m = new Mat(matHeight, matWidth, CV_32F, output.ptr(0, i));
            if (resizeMat)
                resize(m, m, inputSize);
            mats[i] = m;
        }

        return mats;
    }

    // debug
    protected void printLayerNames() {
        StringVector layerNames = net.getLayerNames();
        for (int i = 0; i < layerNames.size(); i++) {
            System.out.println("#" + i + ": " + layerNames.get(i).getString());
        }
    }

    protected void storeHeatMap(String path, Mat image) {
        Mat colorMap = new Mat();
        image.convertTo(colorMap, CV_8U, scale, -mean);
        applyColorMap(colorMap, colorMap, COLORMAP_JET);
        imwrite(path, colorMap);
    }

    public Path getModelPath() {
        return modelPath;
    }

    public Net getNet() {
        return net;
    }

    public double getMean() {
        return mean;
    }

    public double getScale() {
        return scale;
    }
}
