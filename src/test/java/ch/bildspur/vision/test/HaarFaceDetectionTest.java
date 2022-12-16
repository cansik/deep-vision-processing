package ch.bildspur.vision.test;


import ch.bildspur.vision.CascadeClassifierNetwork;
import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class HaarFaceDetectionTest extends PApplet {

    public static void main(String... args) {
        HaarFaceDetectionTest sketch = new HaarFaceDetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    CascadeClassifierNetwork network;
    List<ObjectDetectionResult> detections;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        network = vision.createCascadeFrontalFace();

        println("loading model...");
        network.setup();

        // apply settings
        network.setScaleFactor(1.01);
        network.setMinNeighbors(3);

        println("inferencing...");
        detections = network.run(testImage);
        println("done!");

        for (ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
        }

        println("found " + detections.size() + " faces!");
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

        surface.setTitle("Face Detection Test - FPS: " + Math.round(frameRate));
    }
}
