package ch.bildspur.vision.test;


import ch.bildspur.vision.DORNDepthEstimationNetwork;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.result.ImageResult;
import processing.core.PApplet;
import processing.core.PImage;

public class DORNDepthEstimationTest extends PApplet {

    public static void main(String... args) {
        DORNDepthEstimationTest sketch = new DORNDepthEstimationTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 480);
    }

    PImage groundTruth;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    DORNDepthEstimationNetwork network;
    ImageResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        groundTruth = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        network = vision.createDORNDepthEstimation();

        println("loading model...");
        network.setup();

        println("inferencing...");
        result = network.run(groundTruth);
        println("done!");

        noLoop();
    }

    public void draw() {
        background(55);
        image(groundTruth, 0, 0);
        image(result.getImage(), 640, 0);
        surface.setTitle("DORN Depth Estimation - FPS: " + Math.round(frameRate));
    }
}
