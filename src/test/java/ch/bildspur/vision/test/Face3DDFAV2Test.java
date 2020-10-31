package ch.bildspur.vision.test;


import ch.bildspur.vision.*;
import ch.bildspur.vision.result.FacialLandmarkResult;
import ch.bildspur.vision.result.KeyPointResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import processing.core.PApplet;
import processing.core.PImage;

public class Face3DDFAV2Test extends PApplet {

    public static void main(String... args) {
        Face3DDFAV2Test sketch = new Face3DDFAV2Test();
        sketch.runSketch();
    }

    public void settings() {
        size(480, 720);
    }

    PImage testImage;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    ULFGFaceDetectionNetwork faceNetwork;
    Face3DDFAV2Network net;

    ResultList<ObjectDetectionResult> detections;
    ResultList<FacialLandmarkResult> markedFaces;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/pexels-photo-3852038.jpeg"));

        println("creating network...");
        faceNetwork = vision.createULFGFaceDetectorRFB640();
        net = vision.create3DDFAV2();

        println("loading model...");
        faceNetwork.setup();
        net.setup();

        print("detect faces...");
        detections = faceNetwork.run(testImage);
        println("done!");

        // scale width of face detection (better for emotions);
        for (ObjectDetectionResult face : detections) {
            face.scale(1.0f, 1.0f);
        }

        print("detecting landmarks...");
        markedFaces = net.runByDetections(testImage, detections);
        println("done!");

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            System.out.println(face.getClassName() + "\t[" + face.getConfidence() + "]");
        }

        println("found " + detections.size() + " faces!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            FacialLandmarkResult landmarks = markedFaces.get(i);

            noFill();
            strokeWeight(2f);
            stroke(200, 80, 100);
            rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());

            noStroke();
            fill(100, 80, 200);
            for (int j = 0; j < landmarks.getKeyPoints().size(); j++) {
                KeyPointResult kp = landmarks.getKeyPoints().get(j);
                ellipse(kp.getX(), kp.getY(), 5, 5);
            }
        }

        surface.setTitle("3DDFAV2 Test FPS: " + Math.round(frameRate));
    }
}
