package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.FERPlusEmotionNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import processing.core.PApplet;
import processing.core.PImage;

public class FERPlusEmotionTest extends PApplet {

    public static void main(String... args) {
        FERPlusEmotionTest sketch = new FERPlusEmotionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(256, 256);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    FERPlusEmotionNetwork network;
    ClassificationResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/happy.png"));

        println("creating network...");
        network = vision.createFERPlusEmotionClassifier();

        println("loading model...");
        network.setup();

        println("inferencing...");
        result = network.run(testImage);
        println("done!");

        println("detected: " + result.getClassName() + " with " + Math.round(100f * result.getConfidence()) + "% confidence!");
    }

    public void draw() {
        background(55);
        image(testImage, 0, 0, width, height);
        surface.setTitle("FER+ Emotion Classifier - FPS: " + Math.round(frameRate));
    }
}
