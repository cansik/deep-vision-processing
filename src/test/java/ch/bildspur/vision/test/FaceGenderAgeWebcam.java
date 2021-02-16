package ch.bildspur.vision.test;


import ch.bildspur.video.Capture;
import ch.bildspur.vision.AgeNetwork;
import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.GenderNetwork;
import ch.bildspur.vision.ULFGFaceDetectionNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import processing.core.PApplet;
import processing.core.PImage;

public class FaceGenderAgeWebcam extends PApplet {

    public static void main(String... args) {
        FaceGenderAgeWebcam sketch = new FaceGenderAgeWebcam();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    Capture cam;

    DeepVision vision = new DeepVision(this);
    ULFGFaceDetectionNetwork faceNetwork;
    GenderNetwork genderNetwork;
    AgeNetwork ageNetwork;

    ResultList<ObjectDetectionResult> detections;
    ResultList<ClassificationResult> genders;
    ResultList<ClassificationResult> ages;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        println("creating network...");
        faceNetwork = vision.createULFGFaceDetectorRFB640();
        genderNetwork = vision.createGenderClassifier();
        ageNetwork = vision.createAgeClassifier();

        println("loading model...");
        faceNetwork.setup();
        genderNetwork.setup();
        ageNetwork.setup();

        // setup camera
        cam = new Capture(this, 640, 480);
        cam.start();
    }

    public void draw() {
        background(55);

        if(cam.available()) {
            cam.read();
        }

        image(cam, 0, 0);

        // detect
        detections = faceNetwork.run(cam);

        // scale width of face detection (better for emotions);
        for (ObjectDetectionResult face : detections) {
            face.scale(1.4f, 1.0f);
        }

        genders = genderNetwork.runByDetections(cam, detections);
        ages = ageNetwork.runByDetections(cam, detections);

        // display
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
