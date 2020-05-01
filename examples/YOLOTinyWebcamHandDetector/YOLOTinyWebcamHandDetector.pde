import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.video.Capture;

Capture cam;
PImage inputImage;

DeepVision deepVision = new DeepVision(this);
YOLONetwork yolo;
ResultList<ObjectDetectionResult> detections;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);

  println("creating model...");
  yolo = deepVision.createCrossHandDetectorTinyPRN();
  yolo.setConfidenceThreshold(0.3);

  println("loading yolo model...");
  yolo.setup();

  String[] cams = Capture.list();
  cam = new Capture(this, 640, 480, cams[1]);
  cam.start();
  
  inputImage = new PImage(320, 240, RGB);
}

public void draw() {
  if (cam.available()) {
    cam.read();
  } else {
    return;
  }
  
  background(55);

  image(cam, 0, 0);
  inputImage.copy(cam, 0, 0, cam.width, cam.height, 0, 0, inputImage.width, inputImage.height);

  detections = yolo.run(inputImage);
  
  scale(2);
  for (ObjectDetectionResult detection : detections) {
    noFill();
    strokeWeight(2f);
    stroke((int)(360.0 / yolo.getLabels().size() * detection.getClassId()), 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

    fill(0);
    text(nf(detection.getConfidence(), 0, 2), detection.getX(), detection.getY());
  }

  surface.setTitle("Webcam YOLO Test - FPS: " + Math.round(frameRate));
}
