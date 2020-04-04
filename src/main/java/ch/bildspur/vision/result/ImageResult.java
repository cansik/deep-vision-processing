package ch.bildspur.vision.result;

import processing.core.PImage;

public class ImageResult implements NetworkResult {
    private PImage image;

    public ImageResult(PImage image) {
        this.image = image;
    }

    public PImage getImage() {
        return image;
    }
}
