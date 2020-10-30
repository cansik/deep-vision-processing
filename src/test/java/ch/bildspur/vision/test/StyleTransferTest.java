package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.StyleTransferNetwork;
import ch.bildspur.vision.dependency.Repository;
import ch.bildspur.vision.result.ImageResult;
import processing.core.PApplet;
import processing.core.PImage;

public class StyleTransferTest extends PApplet {

    public static void main(String... args) {
        StyleTransferTest sketch = new StyleTransferTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 480);
    }

    PImage groundTruth;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    StyleTransferNetwork network;
    ImageResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        groundTruth = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        network = vision.createStyleTransfer(Repository.InstanceNormCandy);

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
        surface.setTitle("Style Transfer - FPS: " + Math.round(frameRate));
    }
}
