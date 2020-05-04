package ch.bildspur.vision.result;

import org.bytedeco.opencv.opencv_core.Mat;

public class ObjectSegmentationResult extends ObjectDetectionResult {
    private Mat mask;

    public ObjectSegmentationResult(int classId, String className, float confidence, int x, int y, int width, int height, Mat mask) {
        super(classId, className, confidence, x, y, width, height);
        this.mask = mask;
    }

    public Mat getMask() {
        return mask;
    }
}
