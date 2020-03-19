package ch.bildspur.vision.network;

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

public abstract class PoseNetwork<R> extends DeepNeuralNetwork<R> implements NetworkFactory {
    private Path modelPath;
    protected Net net;

    private final int inputHeight;
    private final int inputWidth;

    protected Scalar meanScalar;
    protected double scale;

    public PoseNetwork(Path modelPath, int inputWidth, int inputHeight, double scale, double mean) {
        this(modelPath, inputWidth, inputHeight, scale, new Scalar(mean, mean, mean, 0.0));
    }

    public PoseNetwork(Path modelPath, int inputWidth, int inputHeight, double scale, Scalar meanScalar) {
        this.modelPath = modelPath;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.scale = scale;
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
        image.convertTo(colorMap, CV_8U, 256.0, -128);
        applyColorMap(colorMap, colorMap, COLORMAP_JET);
        imwrite(path, colorMap);
    }

    protected float getProbability(double probability) {
        return (float) (probability);
    }

    public Path getModelPath() {
        return modelPath;
    }

    public Net getNet() {
        return net;
    }

    public Scalar getMean() {
        return meanScalar;
    }

    public double getScale() {
        return scale;
    }
}
