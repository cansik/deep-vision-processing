package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.MaskRCNN;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ObjectSegmentationResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

import java.util.List;

public class MaskRCNNDetectionTest extends PApplet {

    public static void main(String... args) {
        MaskRCNNDetectionTest sketch = new MaskRCNNDetectionTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    PImage testImage;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    MaskRCNN rcnn;
    List<ObjectSegmentationResult> detections;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/office.jpg"));

        println("creating network...");
        rcnn = vision.createMaskRCNN();

        println("loading model...");
        rcnn.setup();

        //yolo.setConfidenceThreshold(0.3f);

        println("inferencing...");
        detections = rcnn.run(testImage);
        println("done!");

        float confidenceSum = 0;
        for (ObjectDetectionResult detection : detections) {
            System.out.println(detection.getClassName() + "\t[" + detection.getConfidence() + "]");
            confidenceSum += detection.getConfidence();
        }

        println("found " + detections.size() + " objects. avg conf: " + nf(confidenceSum / detections.size(), 0, 2));
        noLoop();
    }

    public void draw() {
        background(55);

        blendMode(BLEND);
        tint(255, 255);
        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        for (ObjectSegmentationResult detection : detections) {
            // display rect
            int c = color(round(360.0f * (float) detection.getClassId() / rcnn.getLabels().size()), 75, 100);

            stroke(c);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

            textSize(15);
            text(detection.getClassName(), detection.getX(), detection.getY());

            // display mask
            Mat cvMask = detection.getMask();
            PImage mask = new PImage(cvMask.size().width(), cvMask.size().height(), PConstants.RGB);
            CvProcessingUtils.toPImage(detection.getMask(), mask);

            blendMode(SCREEN);
            tint(c, 200);
            image(mask, detection.getX(), detection.getY());
        }

        surface.setTitle("MaskRCNN Test - FPS: " + Math.round(frameRate));
    }
}
