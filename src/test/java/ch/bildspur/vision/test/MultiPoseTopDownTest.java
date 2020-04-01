package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.SSDMobileNetwork;
import ch.bildspur.vision.SingleHumanPoseNetwork;
import ch.bildspur.vision.result.HumanPoseResult;
import ch.bildspur.vision.result.KeyPointResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class MultiPoseTopDownTest extends PApplet {

    public static void main(String... args) {
        MultiPoseTopDownTest sketch = new MultiPoseTopDownTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 942, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    SSDMobileNetwork network;
    SingleHumanPoseNetwork pose;

    List<ObjectDetectionResult> detections;
    List<HumanPoseResult> humanPoseResults = new ArrayList<>();

    float scale = 1.0f;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/sport.jpg"));

        println("creating network...");
        network = vision.createMobileNetV2();
        pose = vision.createSingleHumanPoseEstimation();

        println("loading model...");
        network.setup();
        pose.setup();

        print("inferencing detections...");
        detections = network.run(testImage).stream()
                .filter(e -> e.getClassName().equals("person"))
                .collect(Collectors.toList());
        println("done!");

        println("detected " + detections.size() + " poses.");

        print("inferencing poses...");
        humanPoseResults = pose.runByDetections(testImage, detections);
        println("done!");
    }

    public void draw() {
        background(55);

        testImage.resize(0, height);
        image(testImage, 0, 0);

        // draw result
        scale(1f / scale);

        for (int i = 0; i < detections.size(); i++) {
            noFill();
            strokeWeight(2f);

            ObjectDetectionResult detection = detections.get(i);
            HumanPoseResult pose = humanPoseResults.get(i);

            stroke(200, 80, 100);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

            stroke(300, 80, 100);

            pushMatrix();
            translate(detection.getX(), detection.getY());
            drawHuman(pose);
            popMatrix();
        }

        surface.setTitle("Pose Estimation Test - FPS: " + Math.round(frameRate));
    }

    private void drawHuman(HumanPoseResult human) {
        // draw human
        connect(human.getLeftAnkle(),
                human.getLeftKnee(),
                human.getLeftHip(),
                human.getLeftShoulder(),
                human.getLeftElbow(),
                human.getLeftWrist());

        connect(human.getRightAnkle(),
                human.getRightKnee(),
                human.getRightHip(),
                human.getRightShoulder(),
                human.getRightElbow(),
                human.getRightWrist());

        connect(human.getLeftHip(), human.getRightHip());
        connect(human.getLeftShoulder(), human.getRightShoulder());

        connect(human.getLeftShoulder(), human.getNose(), human.getRightShoulder());
        connect(human.getLeftEar(), human.getLeftEye(), human.getRightEye(), human.getRightEar());
        connect(human.getLeftEye(), human.getNose(), human.getRightEye());

        // draw points
        int i = 0;
        stroke(0, 50, 100);
        for (KeyPointResult point : human.getKeyPoints()) {
            fill(0, 0, 0);
            ellipse(point.getX(), point.getY(), 10, 10);

            fill(0, 50, 100);
            textSize(12);
            text(i + " (" + nf(point.getProbability(), 0, 2) + ")", point.getX() + 5, point.getY());
            i++;
        }
    }

    private void connect(KeyPointResult... keyPoints) {
        for (int i = 0; i < keyPoints.length - 1; i++) {
            KeyPointResult a = keyPoints[i];
            KeyPointResult b = keyPoints[i + 1];

            line(a.getX(), a.getY(), b.getX(), b.getY());
        }
    }
}
