package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.result.FacialLandmarkResult;
import ch.bildspur.vision.result.KeyPointResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.FacemarkLBF;
import processing.core.PImage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static ch.bildspur.vision.util.CvProcessingUtils.createValidROI;

public class FacemarkLBFNetwork extends BaseNeuralNetwork<FacialLandmarkResult> implements MultiProcessingNetwork<FacialLandmarkResult> {
    private Path model;
    private FacemarkLBF net;

    public FacemarkLBFNetwork(Path model) {
        this.model = model;
    }

    @Override
    public boolean setup() {
        net = FacemarkLBF.create();
        net.loadModel(model.toAbsolutePath().toString());
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
        Point2fVectorVector extractedFaces = new Point2fVectorVector();

        for (ObjectDetectionResult detection : detections) {
            rois.push_back(createValidROI(frame.size(), detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight()));
        }

        net.fit(frame, rois, extractedFaces);

        // extract
        for (int f = 0; f < extractedFaces.size(); f++) {
            Point2fVector faceExtraction = extractedFaces.get(f);
            List<KeyPointResult> result = new ArrayList<>();
            for (int i = 0; i < faceExtraction.size(); i++) {
                Point2f keyPoint = faceExtraction.get(i);
                result.add(new KeyPointResult(i, Math.round(keyPoint.x()), Math.round(keyPoint.y()), -1.0f));
            }
            results.add(new FacialLandmarkResult(result));
        }

        return results;
    }
}
