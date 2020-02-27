package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.SingleHumanPoseEstimationNetwork;
import ch.bildspur.vision.result.HumanPoseResult;
import ch.bildspur.vision.result.KeyPointResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class PoseEstimationTest extends PApplet {

    public static void main(String... args) {
        PoseEstimationTest sketch = new PoseEstimationTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);

    SingleHumanPoseEstimationNetwork pose;
    HumanPoseResult result;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/pose.jpg"));

        println("creating network...");
        pose = vision.createSingleHumanPoseEstimation();

        println("loading model...");
        pose.setup();

        println("inferencing...");
        result = pose.run(testImage);
        println("done!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        // draw result
        stroke(50, 80, 100);
        List<KeyPointResult> pts = result.getKeyPoints();
        for (int i = 0; i < pts.size() - 1; i++) {
            stroke(round(360.0f * (float) i / pts.size()), 75, 100);
            KeyPointResult a = pts.get(i);
            KeyPointResult b = pts.get(i + 1);

            if (a.getProbability() > 0.0 && b.getProbability() > 0.0) {
                line(a.getX(), a.getY(), b.getX(), b.getY());
            }
        }

        noFill();
        strokeWeight(2f);

        surface.setTitle("Pose Estimation Test - FPS: " + Math.round(frameRate));
    }
}
