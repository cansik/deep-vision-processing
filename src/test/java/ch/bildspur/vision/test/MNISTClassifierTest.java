package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.MNISTClassificationNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import processing.core.PApplet;
import processing.core.PImage;

public class MNISTClassifierTest extends PApplet {

    public static void main(String... args) {
        MNISTClassifierTest sketch = new MNISTClassifierTest();
        sketch.runSketch();
    }

    public void settings() {
        size(256, 256, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    MNISTClassificationNetwork network;
    ClassificationResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/number2.png"));

        println("creating network...");
        network = vision.createMNISTClassifier();

        println("loading model...");
        network.setup();

        println("inferencing...");
        result = network.run(testImage);
        println("done!");

        println("detected: " + result.getClassId() + " with " + Math.round(100f * result.getConfidence()) + "% confidence!");
    }

    public void draw() {
        background(55);
        image(testImage, 0, 0, width, height);
        surface.setTitle("MNIST Classifier - FPS: " + Math.round(frameRate));
    }
}
