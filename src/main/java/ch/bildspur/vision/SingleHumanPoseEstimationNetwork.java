package ch.bildspur.vision;

import ch.bildspur.vision.result.HumanPoseResult;
import ch.bildspur.vision.result.KeyPointResult;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class SingleHumanPoseEstimationNetwork extends DeepNeuralNetwork<HumanPoseResult> {
    private Path modelPath;
    private Net net;

    private final int inputHeight = 384;
    private final int inputWidth = 288;

    public SingleHumanPoseEstimationNetwork(Path modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(modelPath.toAbsolutePath().toString());

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    HumanPoseResult run(Mat frame) {
        Size inputSize = new Size(inputWidth, inputHeight);
        Scalar zeroScalar = new Scalar(0.0, 0.0, 0.0, 0.0);
        Mat inputBlob = blobFromImage(frame, 1 / 255.0, inputSize, zeroScalar, true, false, CV_32F);

        // inferencing
        net.setInput(inputBlob);
        Mat output = net.forward();

        Mat[] heatMaps = splitNetOutputBlobToParts(output, frame.size());

        // read maximum keypoint
        List<KeyPointResult> keyPoints = new ArrayList<>();
        for (int i = 0; i < heatMaps.length; i++) {
            keyPoints.add(extractKeyPoint(i, heatMaps[i]));
        }

        return new HumanPoseResult(keyPoints);
    }

    private KeyPointResult extractKeyPoint(int index, Mat probMap) {
        // get maximum point
        Point maxPoint = new Point(1);
        DoublePointer probability = new DoublePointer(1);

        // Get the value and location of the maximum score
        minMaxLoc(probMap, null, probability, null, maxPoint, null);

        return new KeyPointResult(index, maxPoint.x(), maxPoint.y(), (float) probability.get());
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
