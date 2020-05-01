package ch.bildspur.vision.result;

public class ObjectSegmentationResult extends ObjectDetectionResult {

    public ObjectSegmentationResult(int classId, String className, float confidence, int x, int y, int width, int height) {
        super(classId, className, confidence, x, y, width, height);
    }
}
