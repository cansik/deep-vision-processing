package ch.bildspur.vision.network;

import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.util.ArrayList;
import java.util.List;

import static ch.bildspur.vision.util.CvProcessingUtils.createValidROI;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public abstract class ClassificationNetwork extends LabeledNetwork<ClassificationResult> implements NetworkFactory {
    private Net net;

    private int width;
    private int height;

    private boolean convertToGrayScale;

    private float scaleFactor;
    private Scalar mean;
    private boolean swapRB;
    private boolean crop;

    private float confidenceScale;

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

        return new ClassificationResult(index, getLabelOrId(index), probability);
    }

    public List<ClassificationResult> runByDetections(PImage image, List<ObjectDetectionResult> detections) {
        Mat frame = convertToMat(image);
        return runByDetections(frame, detections);
    }

    public List<ClassificationResult> runByDetections(Mat frame, List<ObjectDetectionResult> detections) {
        List<ClassificationResult> results = new ArrayList<>();

        for (ObjectDetectionResult detection : detections) {
            Mat roi = new Mat(frame, createValidROI(frame.size(), detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight()));
            ClassificationResult result = run(roi);
            results.add(result);
            roi.release();
        }

        return results;
    }
}
