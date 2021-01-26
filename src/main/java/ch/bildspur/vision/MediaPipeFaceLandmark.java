package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.result.FacialLandmarkResult;
import ch.bildspur.vision.result.KeyPointResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
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
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;

public class MediaPipeFaceLandmark extends BaseNeuralNetwork<FacialLandmarkResult> implements MultiProcessingNetwork<FacialLandmarkResult> {
    private Path model;
    private Path config;
    private Net net;

    public MediaPipeFaceLandmark(Path model, Path config) {
        this.model = model;
        this.config = config;
    }

    @Override
    public boolean setup() {
        net = readNetFromTensorflow(model.toAbsolutePath().toString(), config.toAbsolutePath().toString());
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

        for (ObjectDetectionResult detection : detections) {
            Rect roiRect = createValidROI(frame.size(), detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
            Mat roi = new Mat(frame, roiRect);

            // extract
            Mat inputBlob = blobFromImage(roi,
                    1 / 255.0,
                    new Size(192, 192),
                    new Scalar(0.0),
                    false, false, CV_32F);

            // set input
            net.setInput(inputBlob);

            Mat out = net.forward();

            roi.release();

            results.add(new FacialLandmarkResult(new ArrayList<>()));
        }

        return results;
    }
}
