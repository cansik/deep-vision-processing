package ch.bildspur.vision.test;


import ch.bildspur.video.Capture;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.MidasNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.test.tools.StopWatch;
import processing.core.PApplet;
import processing.core.PImage;

public class MidasWebcamTest extends PApplet {

    public static void main(String... args) {
        MidasWebcamTest sketch = new MidasWebcamTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 480);
    }

    Capture cam;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    MidasNetwork network;
    ImageResult result;

    StopWatch watch = new StopWatch();

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        println("creating network...");
        network = vision.createMidasNetworkSmall();

        println("loading model...");
        network.setup();

        // setup camera
        cam = new Capture(this, 640, 480);
        cam.start();
    }

    public void draw() {
        background(55);

        if(cam.available()) {
            cam.read();
        }

        println("inferencing...");
        watch.start();
        result = network.run(cam);
        watch.stop();
        println("done!");

        image(cam, 0, 0);
        image(result.getImage(), 640, 0);
        surface.setTitle("Midas Network Test - FPS: " + Math.round(frameRate));
    }
}
