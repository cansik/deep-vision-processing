package ch.bildspur.vision.network;

import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;

public abstract class ObjectDetectionNetwork extends LabeledNetwork<ResultList<ObjectDetectionResult>> {
    private float confidenceThreshold = 0.5f;

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(float confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
}
