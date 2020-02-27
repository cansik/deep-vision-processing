import ch.bildspur.vision.*;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.video.Capture;

import java.util.List;

Capture cam;

DeepVision deepVision = new DeepVision();
YOLONetwork yolo;
List<ObjectDetectionResult> detections;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);

  println("creating model...");
  yolo = deepVision.createYOLOv3Tiny();

  println("loading yolo model...");
  yolo.setup();

  println("listing cameras...");
  String[] cameras = Capture.list();

  if (cameras.length == 0) {
    println("There are no cameras available for capture.");
    exit();
  } else {
    println("Available cameras:");
    for (int i = 0; i < cameras.length; i++) {
      println(cameras[i]);
    }

    // The camera can be initialized directly using an
    // element from the array returned by list():
    cam = new Capture(this, cameras[0]);
    cam.start();
  }
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
    stroke((int)(360.0 / yolo.getNames().size() * detection.getClassId()), 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
  }

  surface.setTitle("Webcam YOLO Test - FPS: " + Math.round(frameRate));
}
