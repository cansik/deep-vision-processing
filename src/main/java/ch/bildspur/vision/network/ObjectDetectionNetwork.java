package ch.bildspur.vision.network;

import ch.bildspur.vision.result.ObjectDetectionResult;

import java.util.List;

public abstract class ObjectDetectionNetwork extends LabeledNetwork<List<ObjectDetectionResult>> {
    private float confidenceThreshold = 0.5f;

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(float confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
}
