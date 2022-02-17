package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.result.FacialLandmarkResult;
import ch.bildspur.vision.result.KeyPointResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_face.FacemarkLBF;
import processing.core.PImage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static ch.bildspur.vision.util.CvProcessingUtils.createValidROI;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

public class Face3DDFAV2Network extends BaseNeuralNetwork<FacialLandmarkResult> implements MultiProcessingNetwork<FacialLandmarkResult> {
    private Path model;
    private Net net;

    public Face3DDFAV2Network(Path model) {
        this.model = model;
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(model.toAbsolutePath().toString());

        if (DeepVision.ENABLE_CUDA_BACKEND) {
            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
        }

        return true;
    }

    @Override
    public FacialLandmarkResult run(Mat frame) {
        // tries to detect on full frame
        return runByDetections(frame, new ResultList<>(Collections.singletonList(
                new ObjectDetectionResult(0, "face", 1.0f, 0, 0, frame.size().width(), frame.size().height())
        ))).get(0);
    }

    public ResultList<FacialLandmarkResult> runByDetections(PImage image, ResultList<ObjectDetectionResult> detections) {
        Mat frame = convertToMat(image);
        return runByDetections(frame, detections);
    }

    public ResultList<FacialLandmarkResult> runByDetections(Mat frame, ResultList<ObjectDetectionResult> detections) {
        ResultList<FacialLandmarkResult> results = new ResultList<>();
        RectVector rois = new RectVector();

        for (ObjectDetectionResult detection : detections) {
            rois.push_back(createValidROI(frame.size(), detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight()));
        }

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame, 128.0, new Size(120, 120), Scalar.all(-127.5), true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // print output layers
        System.out.println("DEBUG:");
        StringVector layers = net.getUnconnectedOutLayersNames();
        for(int i = 0; i < layers.size(); i++) {
            String name = layers.get(i).getString();
            System.out.println(i + ": " + name);
        }

        // run detection
        Mat out = net.forward();

        // currently only detects the 3DMM regression parameters

        return results;
    }

    public Net getNet() {
        return net;
    }
}
