package ch.bildspur.vision.pipeline;

import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.DetectionPipelineResult;
import ch.bildspur.vision.result.NetworkResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

public class ObjectDetectionPipeline<R extends NetworkResult> extends NetworkPipeline<ResultList<DetectionPipelineResult<R>>> {
    private ObjectDetectionNetwork detectionNetwork;
    private MultiProcessingNetwork<R>[] postProcessors;

    public ObjectDetectionPipeline(ObjectDetectionNetwork detectionNetwork, MultiProcessingNetwork<R>... postProcessors) {
        this.detectionNetwork = detectionNetwork;
        this.postProcessors = postProcessors;
    }

    @Override
    public boolean setup() {
        // todo: implement check or remove it from setup
        detectionNetwork.setup();

        for (MultiProcessingNetwork<R> network : postProcessors) {
            network.setup();
        }

        return true;
    }

    @Override
    public ResultList<DetectionPipelineResult<R>> run(Mat frame) {
        ResultList<DetectionPipelineResult<R>> results = new ResultList<>();

        // detect objects
        List<ObjectDetectionResult> detections = detectionNetwork.run(frame);

        // run post processors on detections
        for (ObjectDetectionResult detection : detections) {

            for (MultiProcessingNetwork<R> network : postProcessors) {

            }
        }

        return results;
    }
}
