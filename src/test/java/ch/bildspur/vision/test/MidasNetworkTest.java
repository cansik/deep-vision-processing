package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.MidasNetwork;
import ch.bildspur.vision.StyleTransferNetwork;
import ch.bildspur.vision.result.ImageResult;
import processing.core.PApplet;
import processing.core.PImage;

public class MidasNetworkTest extends PApplet {

    public static void main(String... args) {
        MidasNetworkTest sketch = new MidasNetworkTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 480);
    }

    PImage groundTruth;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    MidasNetwork network;
    ImageResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        groundTruth = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        network = vision.createMidasNetwork();

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
        surface.setTitle("Midas Network Test - FPS: " + Math.round(frameRate));
    }
}
