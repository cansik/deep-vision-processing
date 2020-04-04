package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.FERPlusEmotionNetwork;
import ch.bildspur.vision.ULFGFaceDetectionNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import processing.core.PApplet;
import processing.core.PImage;

public class FaceAndEmotionTest extends PApplet {

    public static void main(String... args) {
        FaceAndEmotionTest sketch = new FaceAndEmotionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 720, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    ULFGFaceDetectionNetwork faceNetwork;
    FERPlusEmotionNetwork emotionNetwork;

    ResultList<ObjectDetectionResult> detections;
    ResultList<ClassificationResult> emotions;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/faces.png"));

        println("creating network...");
        faceNetwork = vision.createULFGFaceDetectorRFB640();
        emotionNetwork = vision.createFERPlusEmotionClassifier();

        println("loading model...");
        faceNetwork.setup();
        emotionNetwork.setup();

        print("detect faces...");
        detections = faceNetwork.run(testImage);
        println("done!");

        // scale width of face detection (better for emotions);
        for (ObjectDetectionResult face : detections) {
            face.scale(1.4f, 1.0f);
        }

        print("estimate emotions...");
        emotions = emotionNetwork.runByDetections(testImage, detections);
        println("done!");

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            ClassificationResult emotion = emotions.get(i);

            System.out.println(face.getClassName() + "\t[" + face.getConfidence() + "] is "
                    + emotion.getClassName() + "\t[" + emotion.getConfidence() + "]");
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
            ClassificationResult emotion = emotions.get(i);

            rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
            text(emotion.getClassName(), face.getX(), face.getY());
        }

        surface.setTitle("Face / Emotion - FPS: " + Math.round(frameRate));
    }
}
