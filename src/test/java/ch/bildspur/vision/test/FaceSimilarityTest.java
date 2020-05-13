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
import processing.core.PVector;

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

    float[][] resultMatrix;

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

        print("extract embeddings...");
        faceEmbeddings = openFace.runByDetections(testImage, detections);
        println("done!");

        // show result of face recognition
        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            System.out.println(face.getClassName() + "\t[" + face.getConfidence() + "]");
        }

        println("found " + detections.size() + " faces!");

        // create result matrix
        resultMatrix = new float[detections.size()][detections.size()];

        for(int i = 0; i < faceEmbeddings.size(); i++) {
            for (int j = i + 1; j < faceEmbeddings.size(); j++) {
                float[] vi = faceEmbeddings.get(i).getVector();
                float[] vj = faceEmbeddings.get(j).getVector();

                float distance = euclideanDistance(vi, vj);
                resultMatrix[i][j] = distance;
                resultMatrix[j][i] = distance;
            }
        }

        noLoop();
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        // show result mat
        stroke(30, 80, 20);
        for(int i = 0; i < faceEmbeddings.size(); i++) {
            for (int j = i + 1; j < faceEmbeddings.size(); j++) {
                ObjectDetectionResult fi = detections.get(i);
                ObjectDetectionResult fj = detections.get(j);

                PVector a = new PVector(fi.getCenterX(), fi.getCenterY());
                PVector b = new PVector(fj.getCenterX(), fj.getCenterY());

                float distance = resultMatrix[i][j];
                line(a.x, a.y, b.x, b.y);

                PVector textPos = PVector.lerp(a, b, 0.5f);

                fill(map(distance, 0.0f, 1.0f, 0f, 50f), 80, 100);
                textAlign(CENTER, CENTER);
                textSize(12);
                text("D: " + nf(distance, 0, 2), textPos.x, textPos.y);
            }
        }

        textAlign(LEFT, BOTTOM);
        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult face = detections.get(i);
            VectorResult embedding = faceEmbeddings.get(i);

            noFill();
            strokeWeight(2f);
            stroke(200, 80, 100);
            rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());

            fill(200, 80, 100);
            text("ID: " + i, face.getX(), face.getY());
        }

        surface.setTitle("Face Similarity Test - FPS: " + Math.round(frameRate));
    }

    private float euclideanDistance(float[] a, float[] b) {
        assert a.length == b.length;

        float dist = 0;
        for (int i = 0; i < 40; i++) {
            double c = Math.abs(a[i] - b[i]);
            dist += Math.pow(c, 2);
        }
        return (float) Math.sqrt(dist);
    }
}
