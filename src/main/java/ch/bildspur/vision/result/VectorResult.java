package ch.bildspur.vision.result;

import processing.core.PVector;

public class VectorResult implements NetworkResult {
    private float[] vector;

    public VectorResult(float[] vector) {
        this.vector = vector;
    }

    public float[] getVector() {
        return vector;
    }
}
