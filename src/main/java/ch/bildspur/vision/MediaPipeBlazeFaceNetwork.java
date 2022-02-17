package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import ch.bildspur.vision.util.MathUtils;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_text.FloatVector;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.*;

/**
 * Based on https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/caffe/ultra_face_opencvdnn_inference.py
 * Adapted and improved a lot.
 */
public class MediaPipeBlazeFaceNetwork extends ObjectDetectionNetwork {
    private Path modelPath;
    protected Net net;

    private int width;
    private int height;

    private Scalar imageMean = Scalar.all(127);
    private float imageStd = 128.0f;

    public MediaPipeBlazeFaceNetwork(Path modelPath, int width, int height) {
        this.modelPath = modelPath;
        this.width = width;
        this.height = height;
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(modelPath.toAbsolutePath().toString());

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
                1 / imageStd,
                new Size(width, height),
                imageMean,
                false, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        // extract boxes and scores
        Mat boxesOut = outs.get(0);
        Mat confidencesOut = outs.get(1);

        // boxes
        Mat boxes = boxesOut.reshape(0, boxesOut.size(1));

        // class confidences (BACKGROUND, face)
        Mat confidences = confidencesOut.reshape(0, confidencesOut.size(1));

        return new ResultList<>();
    }

    public Net getNet() {
        return net;
    }
}
