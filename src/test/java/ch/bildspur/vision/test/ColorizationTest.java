package ch.bildspur.vision.test;


import ch.bildspur.vision.ColorizationNetwork;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.StyleTransferNetwork;
import ch.bildspur.vision.result.ImageResult;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

import java.nio.file.Paths;

public class ColorizationTest extends PApplet {

    public static void main(String... args) {
        ColorizationTest sketch = new ColorizationTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 480);
    }

    PImage groundTruth;
    PImage blackWhite;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    ColorizationNetwork network;
    ImageResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        groundTruth = loadImage(sketchPath("data/kalen-emsley-Bkci_8qcdvQ-unsplash.jpg"));
        blackWhite = loadImage(sketchPath("data/kalen-emsley-Bkci_8qcdvQ-unsplash.jpg"));
        blackWhite.filter(PConstants.GRAY);

        println("creating network...");
        network = vision.createColorizationNetwork();

        println("loading model...");
        network.setup();

        println("inferencing...");
        result = network.run(blackWhite);
        println("done!");

        noLoop();
    }

    public void draw() {
        background(55);
        image(groundTruth, 0, 0);
        image(result.getImage(), 640, 0);
        surface.setTitle("Style Transfer - FPS: " + Math.round(frameRate));
    }
}
