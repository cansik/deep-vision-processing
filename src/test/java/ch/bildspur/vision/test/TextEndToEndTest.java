package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.TesseractNetwork;
import ch.bildspur.vision.TextBoxesNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import ch.bildspur.vision.result.TextResult;
import processing.core.PApplet;
import processing.core.PImage;

public class TextEndToEndTest extends PApplet {

    public static void main(String... args) {
        TextEndToEndTest sketch = new TextEndToEndTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    TextBoxesNetwork network;
    TesseractNetwork tesseract;

    ResultList<ObjectDetectionResult> detections;
    ResultList<TextResult> textResults;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/sticker.jpg"));

        println("creating network...");
        network = vision.createTextBoxesDetector();
        tesseract = vision.createTesseractRecognizer();

        println("loading model...");
        network.setup();
        tesseract.setup();

        println("detecting text...");
        detections = network.run(testImage);
        println("done!");

        println("recognizing text...");
        textResults = tesseract.runByDetections(testImage, detections);

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult detection = detections.get(i);
            TextResult textResult = textResults.get(i);

            System.out.println("Text: " + textResult.getText() + "\t[" + detection.getConfidence() + "]");
        }

        println("found " + detections.size() + " texts!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        for (int i = 0; i < detections.size(); i++) {
            ObjectDetectionResult detection = detections.get(i);
            TextResult textResult = textResults.get(i);
            String text = "'" + textResult.getText() + "'";

            noStroke();
            fill(0, 0, 100);
            textSize(20);
            rect(detection.getX(), detection.getY(), textWidth(text), -22);

            fill(200, 80, 30);
            textAlign(LEFT, CENTER);
            text(text, detection.getX(), detection.getY());

            noFill();
            strokeWeight(2f);
            stroke(200, 80, 100);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
        }

        surface.setTitle("Text Detection & Recognition Test - FPS: " + Math.round(frameRate));
    }
}
