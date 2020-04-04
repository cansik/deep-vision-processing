package ch.bildspur.vision.result;

public class ClassificationResult implements NetworkResult {
    private int classId;
    private String className;
    private float confidence;

    public ClassificationResult(int classId, String className, float confidence) {
        this.classId = classId;
        this.className = className;
        this.confidence = confidence;
    }

    public int getClassId() {
        return classId;
    }

    public String getClassName() {
        return className;
    }

    public float getConfidence() {
        return confidence;
    }
}
