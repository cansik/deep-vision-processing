package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.FSRCNNNetwork;
import ch.bildspur.vision.result.ImageResult;
import processing.core.PApplet;
import processing.core.PImage;

public class FSRCNNTest extends PApplet {

    public static void main(String... args) {
        FSRCNNTest sketch = new FSRCNNTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 960, FX2D);
    }

    PImage testImage;
    PImage groundTruth;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    FSRCNNNetwork network;
    ImageResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/office_small.jpg"));
        groundTruth = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        network = vision.createFSCRNN();

        println("loading model...");
        network.setup();

        println("inferencing...");
        result = network.run(testImage);
        println("done!");

        // test
        testImage.resize(640, 0);

        noLoop();
    }

    public void draw() {
        background(55);
        image(testImage, 0, 0);
        image(result.getImage(), 640, 0);
        image(groundTruth, 320, 480);
        surface.setTitle("FSRCNN Super Resolution - FPS: " + Math.round(frameRate));
    }
}
