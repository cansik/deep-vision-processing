package ch.bildspur.vision.network;

import ch.bildspur.vision.result.ObjectSegmentationResult;
import ch.bildspur.vision.result.ResultList;

public abstract class ObjectSegmentationNetwork extends LabeledNetwork<ResultList<ObjectSegmentationResult>> implements ConfidenceThresholdNetwork {
    private float confidenceThreshold = 0.5f;

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(float confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
}
