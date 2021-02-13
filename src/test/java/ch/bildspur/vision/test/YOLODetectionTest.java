package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.YOLONetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.test.tools.StopWatch;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class YOLODetectionTest extends PApplet {

    public static void main(String... args) {
        YOLODetectionTest sketch = new YOLODetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    PImage testImage;
    PImage officeImage;
    PImage sportImage;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    YOLONetwork yolo;
    List<ObjectDetectionResult> detections;

    StopWatch watch = new StopWatch();

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        officeImage = loadImage(sketchPath("data/office.jpg"));
        sportImage = loadImage(sketchPath("data/sport.jpg"));
        testImage = officeImage;

        println("creating network...");
        yolo = vision.createYOLOv4Tiny();

        println("loading model...");
        yolo.setup();

        yolo.setConfidenceThreshold(0.2f);

        println("inferencing...");
        watch.start();
        detections = yolo.run(testImage);
        watch.stop();
        println("done!");

        float confidenceSum = 0;
        for (ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
            confidenceSum += detection.getConfidence();
        }

        println("found " + detections.size() + " objects. avg conf: " + nf(confidenceSum / detections.size(), 0, 2));
    }

    public void draw() {
        background(55);

        testImage = (frameCount % 2 == 0) ? officeImage : sportImage;

        watch.start();
        detections = yolo.run(testImage);
        watch.stop();

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
