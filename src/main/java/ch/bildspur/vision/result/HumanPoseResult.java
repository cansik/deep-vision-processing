package ch.bildspur.vision.result;

import java.util.List;

public class HumanPoseResult {
    private List<KeyPointResult> keyPoints;

    public HumanPoseResult(List<KeyPointResult> keyPoints) {
        this.keyPoints = keyPoints;
    }

    public List<KeyPointResult> getKeyPoints() {
        return keyPoints;
    }

    public KeyPointResult getRightAnkle() {
        return keyPoints.get(0);
    }

    public KeyPointResult getRightKnee() {
        return keyPoints.get(1);
    }

    public KeyPointResult getRightHip() {
        return keyPoints.get(2);
    }

    public KeyPointResult getLeftHip() {
        return keyPoints.get(3);
    }

    public KeyPointResult getLeftKnee() {
        return keyPoints.get(4);
    }

    public KeyPointResult getLeftAnkle() {
        return keyPoints.get(5);
    }

    public KeyPointResult getPel() {
        return keyPoints.get(6);
    }

    public KeyPointResult getSpi() {
        return keyPoints.get(7);
    }

    public KeyPointResult getNeck() {
        return keyPoints.get(8);
    }

    public KeyPointResult getHead() {
        return keyPoints.get(9);
    }

    public KeyPointResult getRightWrist() {
        return keyPoints.get(10);
    }

    public KeyPointResult getRightElbow() {
        return keyPoints.get(11);
    }

    public KeyPointResult getRightShoulder() {
        return keyPoints.get(12);
    }

    public KeyPointResult getLeftShoulder() {
        return keyPoints.get(13);
    }

    public KeyPointResult getLeftElbow() {
        return keyPoints.get(14);
    }

    public KeyPointResult getLeftWrist() {
        return keyPoints.get(15);
    }
}
