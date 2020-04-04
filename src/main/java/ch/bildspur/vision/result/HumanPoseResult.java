package ch.bildspur.vision.result;

import java.util.List;

/**
 * LIP based Human Result (16 Parts)
 */
public class HumanPoseResult implements NetworkResult {
    private List<KeyPointResult> keyPoints;

    public HumanPoseResult(List<KeyPointResult> keyPoints) {
        this.keyPoints = keyPoints;
    }

    public List<KeyPointResult> getKeyPoints() {
        return keyPoints;
    }

    public KeyPointResult getNose() {
        return keyPoints.get(0);
    }

    public KeyPointResult getLeftEye() {
        return keyPoints.get(1);
    }

    public KeyPointResult getRightEye() {
        return keyPoints.get(2);
    }

    public KeyPointResult getLeftEar() {
        return keyPoints.get(3);
    }

    public KeyPointResult getRightEar() {
        return keyPoints.get(4);
    }

    public KeyPointResult getLeftShoulder() {
        return keyPoints.get(5);
    }

    public KeyPointResult getRightShoulder() {
        return keyPoints.get(6);
    }

    public KeyPointResult getLeftElbow() {
        return keyPoints.get(7);
    }

    public KeyPointResult getRightElbow() {
        return keyPoints.get(8);
    }

    public KeyPointResult getLeftWrist() {
        return keyPoints.get(9);
    }

    public KeyPointResult getRightWrist() {
        return keyPoints.get(10);
    }

    public KeyPointResult getLeftHip() {
        return keyPoints.get(11);
    }

    public KeyPointResult getRightHip() {
        return keyPoints.get(12);
    }

    public KeyPointResult getLeftKnee() {
        return keyPoints.get(13);
    }

    public KeyPointResult getRightKnee() {
        return keyPoints.get(14);
    }

    public KeyPointResult getLeftAnkle() {
        return keyPoints.get(15);
    }

    public KeyPointResult getRightAnkle() {
        return keyPoints.get(16);
    }
}
