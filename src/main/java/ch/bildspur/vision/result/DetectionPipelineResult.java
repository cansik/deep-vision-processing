package ch.bildspur.vision.result;

public class DetectionPipelineResult<R extends NetworkResult> implements NetworkResult {
    private ObjectDetectionResult detection;
    private ResultList<R> results;

    public DetectionPipelineResult(ObjectDetectionResult detection, ResultList<R> results) {
        this.detection = detection;
        this.results = results;
    }

    public ObjectDetectionResult getDetection() {
        return detection;
    }

    public ResultList<R> getResults() {
        return results;
    }
}
