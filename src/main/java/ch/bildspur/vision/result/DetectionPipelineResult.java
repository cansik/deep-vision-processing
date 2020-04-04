package ch.bildspur.vision.result;

public class DetectionPipelineResult<R extends NetworkResult> implements NetworkResult {
    private ObjectDetectionResult detection;
    private R[] results;

    public DetectionPipelineResult(ObjectDetectionResult detection, R... results) {
        this.detection = detection;
        this.results = results;
    }

    public ObjectDetectionResult getDetection() {
        return detection;
    }

    public R[] getResults() {
        return results;
    }
}
