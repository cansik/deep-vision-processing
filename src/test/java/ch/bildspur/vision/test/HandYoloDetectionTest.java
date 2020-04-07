package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.YOLONetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class HandYoloDetectionTest extends PApplet {

    public static void main(String... args) {
        HandYoloDetectionTest sketch = new HandYoloDetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    YOLONetwork network;
    List<ObjectDetectionResult> detections;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/hand.jpg"));

        println("creating network...");
        network = vision.createYOLOv3HandDetector();

        println("loading model...");
        network.setup();
        network.setConfidenceThreshold(0.1f);

        println("inferencing...");
        detections = network.run(testImage);
        println("done!");

        for (ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
        }

        println("found " + detections.size() + " hands!");
    }

    public void draw() {
        background(55);

        //detections = network.run(testImage);
        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        stroke(200, 80, 100);
        for (ObjectDetectionResult detection : detections) {
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
        }

        surface.setTitle("Hand Detection Test - FPS: " + Math.round(frameRate));
    }
}
