package ch.bildspur.vision.test;


import ch.bildspur.vision.AgeNetwork;
import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.GenderNetwork;
import ch.bildspur.vision.ULFGFaceDetectionNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import processing.core.PApplet;
import processing.core.PImage;

import java.util.List;

public class FaceGenderAgeTest extends PApplet {

    public static void main(String... args) {
        FaceGenderAgeTest sketch = new FaceGenderAgeTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 720, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    ULFGFaceDetectionNetwork faceNetwork;
    GenderNetwork genderNetwork;
    AgeNetwork ageNetwork;

    List<ObjectDetectionResult> detections;
    List<ClassificationResult> genders;
    List<ClassificationResult> ages;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/faces.png"));

        println("creating network...");
        faceNetwork = vision.createULFGFaceDetectorRFB640();
        genderNetwork = vision.createGenderClassifier();
        ageNetwork = vision.createAgeClassifier();

        println("loading model...");
        faceNetwork.setup();
        genderNetwork.setup();
        ageNetwork.setup();

        print("detect faces...");
        detections = faceNetwork.run(testImage);
        println("done!");

        // scale width of face detection (better for emotions);
        for (ObjectDetectionResult face : detections) {
            face.scale(1.4f, 1.0f);
        }

        print("estimate age and gender...");
        genders = genderNetwork.runByDetections(testImage, detections);
        ages = ageNetwork.runByDetections(testImage, detections);
        println("done!");

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            ClassificationResult gender = genders.get(i);
            ClassificationResult age = ages.get(i);

            System.out.println(face.getClassName() + "\t[" + face.getConfidence() + "] is "
                    + gender.getClassName() + "\t[" + gender.getConfidence() + "] and "
                    + age.getClassName() + "\t[" + age.getConfidence() + "]");
        }

        println("found " + detections.size() + " faces!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        stroke(200, 80, 100);
        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            ClassificationResult gender = genders.get(i);
            ClassificationResult age = ages.get(i);

            rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
            text(gender.getClassName() + ", " + age.getClassName(), face.getX(), face.getY());
        }

        surface.setTitle("Face / Emotion - FPS: " + Math.round(frameRate));
    }
}
