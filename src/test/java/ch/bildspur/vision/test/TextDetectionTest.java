package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.TextBoxesNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class TextDetectionTest extends PApplet {

    public static void main(String... args) {
        TextDetectionTest sketch = new TextDetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    TextBoxesNetwork network;
    List<ObjectDetectionResult> detections;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/sticker.jpg"));

        println("creating network...");
        network = vision.createTextBoxesDetector();

        println("loading model...");
        network.setup();

        //network.setConfidenceThreshold(0.2f);

        println("inferencing...");
        detections = network.run(testImage);
        println("done!");

        for (ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
        }

        println("found " + detections.size() + " texts!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        stroke(200, 80, 100);
        for (ObjectDetectionResult detection : detections) {
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
        }

        surface.setTitle("Text Detection Test - FPS: " + Math.round(frameRate));
    }
}
