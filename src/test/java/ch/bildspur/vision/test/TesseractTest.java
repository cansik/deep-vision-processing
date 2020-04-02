package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.TesseractNetwork;
import ch.bildspur.vision.result.TextResult;
import processing.core.PApplet;
import processing.core.PImage;

public class TesseractTest extends PApplet {

    public static void main(String... args) {
        TesseractTest sketch = new TesseractTest();
        sketch.runSketch();
    }

    public void settings() {
        size(256, 256, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    TesseractNetwork network;
    TextResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/text.png"));

        println("creating network...");
        network = vision.createTesseractRecognizer();

        println("loading model...");
        network.setup();

        println("inferencing...");
        result = network.run(testImage);
        println("done!");

        println("detected: " + result.getText() + " with " + Math.round(100f * result.getProbability()) + "% confidence!");
    }

    public void draw() {
        background(55);
        imageMode(CENTER);
        image(testImage, width / 2, height / 2);
        surface.setTitle("CRNN Text Recognition - FPS: " + Math.round(frameRate));
    }
}
