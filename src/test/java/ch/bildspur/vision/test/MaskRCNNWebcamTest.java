package ch.bildspur.vision.test;


import ch.bildspur.video.Capture;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.MaskRCNN;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ObjectSegmentationResult;
import ch.bildspur.vision.test.tools.StopWatch;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

import java.util.List;

public class MaskRCNNWebcamTest extends PApplet {

    public static void main(String... args) {
        MaskRCNNWebcamTest sketch = new MaskRCNNWebcamTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    Capture cam;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    MaskRCNN rcnn;
    List<ObjectSegmentationResult> detections;

    StopWatch watch = new StopWatch();

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        println("creating network...");
        rcnn = vision.createMaskRCNN();

        println("loading model...");
        rcnn.setup();

        // setup camera
        cam = new Capture(this, 640, 480);
        cam.start();
    }

    public void draw() {
        background(55);

        if(cam.available()) {
            cam.read();
        }

        watch.start();
        detections = rcnn.run(cam);
        watch.stop();


        blendMode(BLEND);
        tint(255, 255);
        image(cam, 0, 0);

        noFill();
        strokeWeight(2f);

        for (ObjectSegmentationResult detection : detections) {
            // display rect
            int c = color(round(360.0f * (float) detection.getClassId() / rcnn.getLabels().size()), 75, 100);

            stroke(c);
            System.out.println("X: " + detection.getX() + " Y: " + detection.getY() + " W: " + detection.getWidth() + " H: " + detection.getHeight());
            //rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

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
