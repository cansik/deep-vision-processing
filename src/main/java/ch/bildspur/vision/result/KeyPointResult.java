package ch.bildspur.vision.result;

public class KeyPointResult implements NetworkResult {
    private final int id;
    private final int x;
    private final int y;
    private final float probability;

    public KeyPointResult(int id, int x, int y, float probability) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.probability = probability;
    }

    public int getId() {
        return id;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public float getProbability() {
        return probability;
    }
}
