package ch.bildspur.vision.test;


import ch.bildspur.vision.CvProcessingUtils;
import ch.bildspur.vision.network.YoloDetection;
import ch.bildspur.vision.network.YoloNetwork;
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
    YoloNetwork yolo;
    List<YoloDetection> detections;

    public void setup() {
        testImage = loadImage(sketchPath("data/test.jpeg"));
        prepared = new PImage(testImage.width, testImage.height, PConstants.RGB);

        yolo = new YoloNetwork();
        yolo.setup();

        detections = yolo.detect(testImage, 0.001f);

        for(YoloDetection detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
        }
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        stroke(0, 255, 100);
        strokeWeight(2f);

        for(YoloDetection detection : detections) {
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
        }

        surface.setTitle("RealSense Processing - FPS: " + Math.round(frameRate));
    }
}
