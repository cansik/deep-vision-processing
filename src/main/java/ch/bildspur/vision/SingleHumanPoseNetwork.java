package ch.bildspur.vision;

import ch.bildspur.vision.result.HumanPoseResult;
import ch.bildspur.vision.result.KeyPointResult;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

public class SingleHumanPoseNetwork extends PoseNetwork<HumanPoseResult> {

    public SingleHumanPoseNetwork(Path modelPath, int inputWidth, int inputHeight) {
        super(modelPath, inputWidth, inputHeight);
    }

    @Override
    public Net createNetwork() {
        return readNetFromONNX(getModelPath().toAbsolutePath().toString());
    }

    @Override
    HumanPoseResult run(Mat frame) {
        Mat[] heatMaps = extractHeatMaps(frame);

        // read maximum keypoint per heatmap
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
}
