package ch.bildspur.vision;

import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.network.MultiProcessor;
import ch.bildspur.vision.network.PoseNetwork;
import ch.bildspur.vision.result.HumanPoseResult;
import ch.bildspur.vision.result.KeyPointResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

public class SingleHumanPoseNetwork extends PoseNetwork<HumanPoseResult> implements MultiProcessingNetwork<HumanPoseResult> {
    private final int pointCount = 17;
    private final MultiProcessor<HumanPoseResult> polyExecutor = new MultiProcessor<>(this);

    public SingleHumanPoseNetwork(Path modelPath) {
        // todo: check if this is really LIP or COCO dataset
        super(modelPath, 288, 384, 58.0, new Scalar(123.675, 116.28, 103.53, 0.0));
    }

    @Override
    public Net createNetwork() {
        return readNetFromONNX(getModelPath().toAbsolutePath().toString());
    }

    @Override
    public HumanPoseResult run(Mat frame) {
        // prepare
        Mat inputBlob = createInputBlob(frame);

        // inference
        getNet().setInput(inputBlob);
        Mat output = net.forward("stage_4_output_1_heatmaps");

        // post-process
        Mat[] heatMaps = splitNetOutputBlobToParts(output, frame.size(), true);

        // read maximum keypoint per heatmap
        List<KeyPointResult> keyPoints = new ArrayList<>();
        for (int i = 0; i < pointCount; i++) {
            keyPoints.add(extractKeyPoint(i, heatMaps[i]));
        }

        return new HumanPoseResult(keyPoints);
    }

    private KeyPointResult extractKeyPoint(int index, Mat probMap) {
        // storeHeatMap("maps/c_" + index + ".bmp", probMap);

        // get maximum point
        Point maxPoint = new Point(1);
        DoublePointer probability = new DoublePointer(1);

        // Get the value and location of the maximum score
        minMaxLoc(probMap, null, probability, null, maxPoint, null);

        return new KeyPointResult(index, maxPoint.x(), maxPoint.y(), getProbability(probability.get()));
    }

    @Override
    public ResultList<HumanPoseResult> runByDetections(PImage image, ResultList<ObjectDetectionResult> detections) {
        return polyExecutor.runByDetections(image, detections);
    }

    @Override
    public ResultList<HumanPoseResult> runByDetections(Mat frame, ResultList<ObjectDetectionResult> detections) {
        return polyExecutor.runByDetections(frame, detections);
    }
}
