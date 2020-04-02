package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.YOLONet;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class YOLODetectionTest extends PApplet {

    public static void main(String... args) {
        YOLODetectionTest sketch = new YOLODetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    YOLONet yolo;
    List<ObjectDetectionResult> detections;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/raum.jpeg"));

        println("creating network...");
        yolo = vision.createYOLOv3();

        println("loading model...");
        yolo.setup();

        yolo.setConfidenceThreshold(0.2f);

        println("inferencing...");
        detections = yolo.run(testImage);
        println("done!");

        for (ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
        }
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        for (ObjectDetectionResult detection : detections) {
            stroke(round(360.0f * (float) detection.getClassId() / yolo.getLabels().size()), 75, 100);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

            textSize(15);
            text(detection.getClassName(), detection.getX(), detection.getY());
        }

        surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
    }
}
