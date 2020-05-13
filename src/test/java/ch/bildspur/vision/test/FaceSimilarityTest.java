package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.FERPlusEmotionNetwork;
import ch.bildspur.vision.OpenFaceNetwork;
import ch.bildspur.vision.ULFGFaceDetectionNetwork;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import ch.bildspur.vision.result.VectorResult;
import processing.core.PApplet;
import processing.core.PImage;

public class FaceSimilarityTest extends PApplet {

    public static void main(String... args) {
        FaceSimilarityTest sketch = new FaceSimilarityTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1000, 720);
    }

    PImage testImage;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    ULFGFaceDetectionNetwork faceNetwork;
    OpenFaceNetwork openFace;

    ResultList<ObjectDetectionResult> detections;
    ResultList<VectorResult> faceEmbeddings;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/children.jpg"));

        println("creating network...");
        faceNetwork = vision.createULFGFaceDetectorRFB640();
        openFace = vision.createOpenFaceNetwork();

        println("loading model...");
        faceNetwork.setup();
        openFace.setup();

        print("detect faces...");
        detections = faceNetwork.run(testImage);
        println("done!");

        // scale width of face detection (better for emotions);
        for (ObjectDetectionResult face : detections) {
            face.scale(1.4f, 1.0f);
            //face.squareByHeight();
        }

        print("estimate emotions...");
        faceEmbeddings = openFace.runByDetections(testImage, detections);
        println("done!");

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            VectorResult embedding = faceEmbeddings.get(i);

            System.out.println(face.getClassName() + "\t[" + face.getConfidence() + "]");
        }

        println("found " + detections.size() + " faces!");
        noLoop();
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        stroke(200, 80, 100);
        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            VectorResult embedding = faceEmbeddings.get(i);

            rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
            //text(emotion.getClassName(), face.getX(), face.getY());
        }

        surface.setTitle("Face Similarity Test - FPS: " + Math.round(frameRate));
    }
}
