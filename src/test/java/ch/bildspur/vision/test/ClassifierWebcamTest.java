package ch.bildspur.vision.test;


import ch.bildspur.video.Capture;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.TensorflowClassifierNetwork;
import ch.bildspur.vision.network.ClassificationNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.test.tools.StopWatch;
import processing.core.PApplet;

import java.nio.file.Paths;
import java.util.List;

public class ClassifierWebcamTest extends PApplet {

    public static void main(String... args) {
        ClassifierWebcamTest sketch = new ClassifierWebcamTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    Capture cam;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    ClassificationNetwork network;
    ClassificationResult result;

    StopWatch watch = new StopWatch();

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        println("Is CUDA Enabled: " + vision.isCUDABackendEnabled());

        println("creating network...");
        network = new TensorflowClassifierNetwork(Paths.get("networks/tool.pb"), 224, 224, "tool", "pen", "none");

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

        watch.start();
        result = network.run(cam);
        watch.stop();

        image(cam, 0, 0);

        noFill();
        strokeWeight(2f);

        textSize(15);
        text("Detected: " + result.getClassName() + " (" + result.getClassId() + ")", 30, 30);

        surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
    }
}
