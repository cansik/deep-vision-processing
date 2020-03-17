package ch.bildspur.vision;

import ch.bildspur.vision.result.HumanPoseResult;
import ch.bildspur.vision.result.KeyPointResult;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.StringVector;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class MultiHumanPoseNetwork extends PoseNetwork<List<HumanPoseResult>> {
    private final int pointCount = 18;
    private final StringVector outputLayerNames = new StringVector("stage_1_output_0_pafs", "stage_1_output_1_heatmaps");

    private double threshold = 0.1f;

    public MultiHumanPoseNetwork(Path modelPath) {
        super(modelPath, 456, 256, 256.0, 128.0);
    }

    @Override
    public Net createNetwork() {
        return readNetFromONNX(getModelPath().toAbsolutePath().toString());
    }

    @Override
    public List<HumanPoseResult> run(Mat frame) {
        // prepare
        Mat inputBlob = createInputBlob(frame);

        // inference
        getNet().setInput(inputBlob);
        MatVector outs = new MatVector(outputLayerNames.size());
        net.forward(outs, outputLayerNames);

        // extract
        Mat poseAffinityFieldOutput = outs.get(0);
        Mat heatMapOutput = outs.get(1);

        // post-process
        Mat[] affinityFields = splitNetOutputBlobToParts(poseAffinityFieldOutput, frame.size(), true);
        Mat[] heatMaps = splitNetOutputBlobToParts(heatMapOutput, frame.size(), true);

        for (int i = 0; i < affinityFields.length; i++) {
            storeHeatMap("maps/a_" + i + ".bmp", affinityFields[i]);
        }

        // save global heat map
        storeHeatMap("maps/global.bmp", heatMaps[heatMaps.length - 1]);

        List<HumanPoseResult> humans = new ArrayList<>();

        for (int i = 0; i < pointCount; i++) {
            List<KeyPointResult> keyPoints = extractKeyPoints(i, heatMaps[i]);
        }

        return humans;
    }

    private List<KeyPointResult> extractKeyPoints(int index, Mat probMap) {
        //storeHeatMap("maps/c_" + index + ".bmp", probMap);

        // smooth prob map and threshold
        Mat smoothProbMap = new Mat();
        GaussianBlur(probMap, smoothProbMap, new Size(3, 3), 0.0, 0.0, 0);

        threshold(smoothProbMap, smoothProbMap, threshold, 255.0, THRESH_BINARY);

        storeHeatMap("maps/c_" + index + ".bmp", smoothProbMap);

        smoothProbMap.convertTo(smoothProbMap, CV_8U);

        //imwrite("maps/k_" + index + ".bmp", smoothProbMap);

        // todo: connected component analysis and select center
        return new ArrayList<>();
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}
