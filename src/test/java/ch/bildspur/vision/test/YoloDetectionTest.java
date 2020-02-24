package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.network.YoloDetection;
import ch.bildspur.vision.network.YoloNetwork;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;
import java.util.Vector;

public class YoloDetectionTest extends PApplet {

    public static void main(String... args) {
        YoloDetectionTest sketch = new YoloDetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;
    YoloNetwork yolo;
    List<YoloDetection> detections;

    public void setup() {
        testImage = loadImage(sketchPath("data/test.jpeg"));

        yolo = new YoloNetwork();
        yolo.setup();

        detections = yolo.detect(testImage, 0.2f);
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        stroke(0, 255, 0);
        strokeWeight(2f);

        for(YoloDetection detection : detections) {
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
        }

        surface.setTitle("RealSense Processing - FPS: " + Math.round(frameRate));
    }
}
