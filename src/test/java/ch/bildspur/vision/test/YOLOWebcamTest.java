package ch.bildspur.vision.test;


import ch.bildspur.video.Capture;
import ch.bildspur.vision.DeepVisionPreview;
import ch.bildspur.vision.YOLONetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.test.tools.StopWatch;
import processing.core.PApplet;

import java.nio.file.Paths;
import java.util.List;

public class YOLOWebcamTest extends PApplet {

    public static void main(String... args) {
        YOLOWebcamTest sketch = new YOLOWebcamTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    Capture cam;

    DeepVisionPreview vision = new DeepVisionPreview(this);
    YOLONetwork yolo;
    List<ObjectDetectionResult> detections;

    StopWatch watch = new StopWatch();

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        println("Is CUDA Enabled: " + vision.isCUDABackendEnabled());

        println("creating network...");
        yolo = vision.createYOLOFastestXL();

        println("loading model...");
        yolo.setup();
        yolo.setConfidenceThreshold(0.2f);

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
        detections = yolo.run(cam);
        watch.stop();

        image(cam, 0, 0);

        noFill();
        strokeWeight(2f);

        for (ObjectDetectionResult detection : detections) {
            stroke(round(360.0f * (float) detection.getClassId() / yolo.getLabels().size()), 75, 100);
            rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

            textSize(15);
            text(detection.getClassName(), detection.getX(), detection.getY());
        }

        surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
    }
}
