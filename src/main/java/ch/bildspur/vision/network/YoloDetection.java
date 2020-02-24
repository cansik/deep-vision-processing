package ch.bildspur.vision.network;

public class YoloDetection {
    private int classId;
    private String className;
    private float confidence;

    // center
    private int x;
    private int y;
    private int width;
    private int height;

    public YoloDetection(int classId, String className, float confidence, int x, int y, int width, int height) {
        this.classId = classId;
        this.className = className;
        this.confidence = confidence;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
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

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}
