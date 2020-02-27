package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.YOLONetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

import java.util.List;

public class YoloDetectionTest extends PApplet {

    public static void main(String... args) {
        YoloDetectionTest sketch = new YoloDetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;
    PImage prepared;

    DeepVision vision = new DeepVision();
    YOLONetwork yolo;
    List<ObjectDetectionResult> detections;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/hk.jpg"));
        prepared = new PImage(testImage.width, testImage.height, PConstants.RGB);

        yolo = vision.createYOLOv3();
        yolo.setup();

        yolo.setConfidenceThreshold(0.2f);
        detections = yolo.run(testImage);

        for(ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
        }
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        for(ObjectDetectionResult detection : detections) {
            stroke((int)(360.0 / yolo.getNames().size() * detection.getClassId()), 80, 100);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
        }

        surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
    }
}
