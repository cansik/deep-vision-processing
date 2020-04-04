import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.video.Capture;

Capture cam;

DeepVision deepVision = new DeepVision(this);
YOLONetwork yolo;
ResultList<ObjectDetectionResult> detections;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);

  println("creating model...");
  yolo = deepVision.createYOLOv3Tiny();

  println("loading yolo model...");
  yolo.setup();

  String[] cams = Capture.list();
  cam = new Capture(this, cams[0]);
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  } else {
    return;
  }

  image(cam, 0, 0);

  yolo.setConfidenceThreshold(0.2f);
  detections = yolo.run(cam);

  noFill();
  strokeWeight(2f);

  for (ObjectDetectionResult detection : detections) {
    stroke((int)(360.0 / yolo.getLabels().size() * detection.getClassId()), 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
  }

  surface.setTitle("Webcam YOLO Test - FPS: " + Math.round(frameRate));
}
