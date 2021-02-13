package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;

public class SSDMobileNetwork extends ObjectDetectionNetwork {

    private Path modelPath;
    private Path configPath;
    protected Net net;

    private int width;
    private int height;

    private float confidenceThreshold;

    public SSDMobileNetwork(Path modelPath, Path configPath, int width, int height, float confidenceThreshold, String... labels) {
        this.modelPath = modelPath;
        this.configPath = configPath;
        this.width = width;
        this.height = height;
        this.confidenceThreshold = confidenceThreshold;

        this.setLabels(labels);
    }

    @Override
    public boolean setup() {
        net = readNetFromTensorflow(
                modelPath.toAbsolutePath().toString(),
                configPath.toAbsolutePath().toString());

        net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
        net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    public ResultList<ObjectDetectionResult> run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1,
                new Size(width, height),
                Scalar.all(0),
                false, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);
        Mat detection = outs.get(0);

        Mat detectionMat = new Mat(detection.size(2), detection.size(3), CV_32F, detection.ptr());

        // extract detections
        ResultList<ObjectDetectionResult> detections = new ResultList<>();
        for (int i = 0; i < detectionMat.rows(); i++) {
            FloatPointer dataPtr = new FloatPointer(detectionMat.row(i).data());

            float confidence = dataPtr.get(2);
            if (confidence < confidenceThreshold) continue;

            int label = Math.round(dataPtr.get(1)) - 1;
            float xLeftBottom = dataPtr.get(3) * frame.cols();
            float yLeftBottom = dataPtr.get(4) * frame.rows();
            float xRightTop = dataPtr.get(5) * frame.cols();
            float yRightTop = dataPtr.get(6) * frame.rows();

            int x = Math.round(xLeftBottom);
            int y = Math.round(yLeftBottom);
            int width = Math.round(xRightTop - xLeftBottom);
            int height = Math.round(yRightTop - yLeftBottom);

            detections.add(new ObjectDetectionResult(label, getLabelOrId(label), confidence,
                    x, y, width, height));
        }

        // todo: implement global nms for object detection algorithms

        return detections;
    }

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(float confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
}
