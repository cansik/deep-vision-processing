package ch.bildspur.vision.result;

public class ObjectDetectionResult extends ClassificationResult {
    // location
    private int x;
    private int y;
    private int width;
    private int height;

    public ObjectDetectionResult(int classId, String className, float confidence, int x, int y, int width, int height) {
        super(classId, className, confidence);

        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
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

    public void scale(float xScale, float yScale) {
        int dx = Math.round(width * xScale) - width;
        int dy = Math.round(height * yScale) - height;

        this.x -= dx;
        this.y -= dy;

        this.width += dx * 2;
        this.height += dy * 2;
    }
}
