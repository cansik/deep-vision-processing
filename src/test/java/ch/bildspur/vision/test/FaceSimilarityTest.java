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

import javax.swing.*;

public class FaceSimilarityTest extends PApplet {

    public static void main(String... args) {
        FaceSimilarityTest sketch = new FaceSimilarityTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 720);
    }

    PImage testImage;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    ULFGFaceDetectionNetwork faceNetwork;
    OpenFaceNetwork openFace;

    ResultList<ObjectDetectionResult> detections;
    ResultList<VectorResult> faceEmbeddings;

    float[][] resultMatrix;

    public void setup() {
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
            face.squareByHeight();
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

        for (int i = 0; i < faceEmbeddings.size(); i++) {
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
        stroke(30, 255, 80);
        for (int i = 0; i < faceEmbeddings.size(); i++) {
            for (int j = i + 1; j < faceEmbeddings.size(); j++) {
                ObjectDetectionResult fi = detections.get(i);
                ObjectDetectionResult fj = detections.get(j);

                PVector a = new PVector(fi.getCenterX(), fi.getCenterY());
                PVector b = new PVector(fj.getCenterX(), fj.getCenterY());

                float distance = resultMatrix[i][j];
                line(a.x, a.y, b.x, b.y);

                PVector textPos = PVector.lerp(a, b, 0.5f);

                fill(colormap(distance));
                textAlign(CENTER, CENTER);
                textSize(15);
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

        fill(255);
        text("Similarity Matrix (lower is better):", 1020, 25);
        drawSimilarityChart(resultMatrix, 1000, 30, 280);

        surface.setTitle("Face Similarity Test - FPS: " + Math.round(frameRate));
    }

    private void drawSimilarityChart(float[][] data, int xOrigin, int yOrigin, int size) {
        float boxSize = size / (data.length + 1f);

        pushMatrix();
        translate(xOrigin, yOrigin);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                float distance = data[i][j];
                float x = boxSize * i + boxSize * 0.5f;
                float y = boxSize * j + boxSize * 0.5f;

                fill(255);
                textAlign(CENTER, CENTER);
                textSize(12);

                if (i == 0)
                    text(j, x - (boxSize * 0.25f), y + boxSize * 0.5f);

                if (j == 0)
                    text(i, x + boxSize * 0.5f, y - (boxSize * 0.25f));

                int colorValue = colormap(distance);
                fill(colorValue);

                if (i == j)
                    noFill();

                stroke(0);
                rect(x, y, boxSize, boxSize);

                // draw value
                fill(brightness(colorValue) > 128 ? 0 : 255);
                text(nf(distance, 0, 2), x + boxSize * 0.5f, y + boxSize * 0.5f);
            }
        }

        // draw map
        pushMatrix();

        float length = boxSize * (data.length + 1f);
        float hl = length * 0.7f;
        float bs = hl / 100f;

        translate((length - hl) * 0.5f, length);

        noStroke();
        for (int i = 0; i < 100; i++) {
            float x = bs * i;

            fill(colormap(i / 100f));
            rect(x, 0, bs, boxSize * 0.2f);

            if (i == 0 || i == 99) {
                fill(255);
                text(i == 0 ? 0 : 1, x, -10);
            }
        }

        popMatrix();
        popMatrix();
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

    private float colormapRed(float x) {
        return (1.0f + 1.0f / 63.0f) * x - 1.0f / 63.0f;
    }

    private float colormapGreen(float x) {
        return -(1.0f + 1.0f / 63.0f) * x + (1.0f + 1.0f / 63.0f);
    }

    int colormap(float x) {
        float r = constrain(colormapRed(x), 0.0f, 1.0f);
        float g = constrain(colormapGreen(x), 0.0f, 1.0f);
        float b = 1.0f;
        return color(r * 255f, g * 255f, b * 255f);
    }
}
