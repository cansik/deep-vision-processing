package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class ObjectDetectionSpeedTest extends PApplet {

    public static void main(String... args) {
        ObjectDetectionSpeedTest sketch = new ObjectDetectionSpeedTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    ObjectDetectionNetwork net;
    List<ObjectDetectionResult> detections;

    float fpsMean = 0;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        net = vision.createMobileNetV2(600);

        println("loading model...");
        net.setup();

        net.setConfidenceThreshold(0.5f);

        println("inferencing...");
        detections = net.run(testImage);
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

        detections = net.run(testImage);
        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        for (ObjectDetectionResult detection : detections) {
            stroke(round(360.0f * (float) detection.getClassId() / net.getLabels().size()), 75, 100);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

            textSize(15);
            text(detection.getClassName(), detection.getX(), detection.getY());
        }

        surface.setTitle("Speed Test - FPS: " + Math.round(frameRate));

        if (frameCount < 5) {
            return;
        } else if (frameCount == 5) {
            println("start measuring...");
        }

        fpsMean += frameRate;

        if (frameCount % 30 == 0) {
            println("Mean (30 Frames): " + nf(fpsMean / 30, 0, 2));
            fpsMean = 0;
            exit();
        }
    }
}
