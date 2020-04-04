package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.pipeline.ObjectDetectionPipeline;
import ch.bildspur.vision.result.DetectionPipelineResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import ch.bildspur.vision.result.TextResult;
import processing.core.PApplet;
import processing.core.PImage;

public class TextEndToEndPipelineTest extends PApplet {

    public static void main(String... args) {
        TextEndToEndPipelineTest sketch = new TextEndToEndPipelineTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    ObjectDetectionPipeline<TextResult> pipeline;
    ResultList<DetectionPipelineResult<TextResult>> results;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/sticker.jpg"));

        println("creating pipeline...");
        pipeline = new ObjectDetectionPipeline<>(
                vision.createTextBoxesDetector(),
                vision.createTesseractRecognizer()
        );

        println("loading pipeline...");
        pipeline.setup();

        println("inference pipeline...");
        results = pipeline.run(testImage);

        for (DetectionPipelineResult<TextResult> result : results) {
            ObjectDetectionResult detection = result.getDetection();
            TextResult textResult = result.getResults().get(0);

            System.out.println("Text: " + textResult.getText() + "\t[" + detection.getConfidence() + "]");
        }

        println("found " + results.size() + " texts!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        for (DetectionPipelineResult<TextResult> result : results) {
            ObjectDetectionResult detection = result.getDetection();
            TextResult textResult = result.getResults().get(0);

            String text = textResult.getText();

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
