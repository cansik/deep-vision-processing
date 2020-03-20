package ch.bildspur.vision.result;

import java.util.List;

public class FacialLandmarkResult {
    private List<KeyPointResult> keyPoints;

    public FacialLandmarkResult(List<KeyPointResult> keyPoints) {
        this.keyPoints = keyPoints;
    }

    public List<KeyPointResult> getKeyPoints() {
        return keyPoints;
    }
}
