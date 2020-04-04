package ch.bildspur.vision.result;

public class TextResult implements NetworkResult {
    private String text;
    private float probability;

    public TextResult(String text, float probability) {
        this.text = text;
        this.probability = probability;
    }

    public String getText() {
        return text;
    }

    public float getProbability() {
        return probability;
    }
}
