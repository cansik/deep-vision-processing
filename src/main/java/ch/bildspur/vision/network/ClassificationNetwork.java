package ch.bildspur.vision.network;

import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public abstract class ClassificationNetwork extends LabeledNetwork<ClassificationResult> implements NetworkFactory, MultiProcessingNetwork<ClassificationResult> {
    private Net net;

    private int width;
    private int height;

    private boolean convertToGrayScale;

    private float scaleFactor;
    private Scalar mean;
    private boolean swapRB;
    private boolean crop;

    private float confidenceScale;

    private final MultiProcessor<ClassificationResult> multiProcessor = new MultiProcessor<>(this);

    public ClassificationNetwork(int width, int height, boolean convertToGrayScale, float scaleFactor, Scalar mean, boolean swapRB, boolean crop, float confidenceScale, String... labels) {
        this.width = width;
        this.height = height;
        this.convertToGrayScale = convertToGrayScale;
        this.scaleFactor = scaleFactor;
        this.mean = mean;
        this.swapRB = swapRB;
        this.crop = crop;
        this.confidenceScale = confidenceScale;

        this.setLabels(labels);
    }

    @Override
    public boolean setup() {
        net = createNetwork();

        DeepVision.enableDesiredBackend(net);

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    public ClassificationResult run(Mat frame) {
        // convert to gray
        if (convertToGrayScale)
            cvtColor(frame, frame, COLOR_RGB2GRAY);

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame, scaleFactor, new Size(width, height), mean, swapRB, crop, CV_32F);

        // set input
        net.setInput(inputBlob);

        // run detection
        Mat out = net.forward();

        // extract result
        Point maxIndexPtr = new Point(1);
        DoublePointer probabilityPtr = new DoublePointer(1);

        minMaxLoc(out, null, probabilityPtr, null, maxIndexPtr, null);

        int index = maxIndexPtr.x();
        float probability = (float) (probabilityPtr.get() / confidenceScale);

        ClassificationResult result = new ClassificationResult(index, getLabelOrId(index), probability);

        // cleanup
        probabilityPtr.releaseReference();
        maxIndexPtr.releaseReference();
        inputBlob.release();
        out.release();

        return result;
    }

    @Override
    public ResultList<ClassificationResult> runByDetections(PImage image, ResultList<ObjectDetectionResult> detections) {
        return multiProcessor.runByDetections(image, detections);
    }

    @Override
    public ResultList<ClassificationResult> runByDetections(Mat frame, ResultList<ObjectDetectionResult> detections) {
        return multiProcessor.runByDetections(frame, detections);
    }

    public Net getNet() {
        return net;
    }
}
