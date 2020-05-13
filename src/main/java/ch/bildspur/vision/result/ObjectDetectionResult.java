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

    public float getCenterX() {
        return x + width * 0.5f;
    }

    public float getCenterY() {
        return y + height * 0.5f;
    }

    public void scale(float xScale, float yScale) {
        int dx = Math.round(width * xScale) - width;
        int dy = Math.round(height * yScale) - height;

        this.x -= dx;
        this.y -= dy;

        this.width += dx * 2;
        this.height += dy * 2;
    }

    public void squareByWidth() {
        this.y = Math.round(y + (height - width) * 0.5f);
        this.height = width;
    }

    public void squareByHeight() {
        this.x = Math.round(x + (width - height) * 0.5f);
        this.width = height;
    }
}
