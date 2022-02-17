package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.indexer.FloatIndexer;
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
    private StringVector outNames;

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

        outNames = net.getUnconnectedOutLayersNames();

        DeepVision.enableDesiredBackend(net);

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
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);
        Mat detection = outs.get(0);

        Mat detectionMat = new Mat(detection.size(2), detection.size(3), CV_32F, detection.ptr());
        FloatIndexer data = detectionMat.createIndexer();

        // extract detections
        ResultList<ObjectDetectionResult> detections = new ResultList<>();
        for (int i = 0; i < detectionMat.rows(); i++) {
            float confidence = data.get(i, 2);
            if (confidence < confidenceThreshold) continue;

            int label = Math.round(data.get(i,1)) - 1;
            float xLeftBottom = data.get(i,3) * frame.cols();
            float yLeftBottom = data.get(i,4) * frame.rows();
            float xRightTop = data.get(i,5) * frame.cols();
            float yRightTop = data.get(i,6) * frame.rows();

            int x = Math.round(xLeftBottom);
            int y = Math.round(yLeftBottom);
            int width = Math.round(xRightTop - xLeftBottom);
            int height = Math.round(yRightTop - yLeftBottom);

            detections.add(new ObjectDetectionResult(label, getLabelOrId(label), confidence,
                    x, y, width, height));
        }

        // todo: implement global nms for object detection algorithms

        inputBlob.release();
        detection.release();
        detectionMat.release();
        outs.releaseReference();

        return detections;
    }

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(float confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }

    public Net getNet() {
        return net;
    }
}
