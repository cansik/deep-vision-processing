import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.video.Capture;

Capture cam;

DeepVision vision = new DeepVision(this);
SSDMobileNetwork network;
ResultList<ObjectDetectionResult> detections;

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  println("creating network...");
  network = vision.createHandDetector();

  println("loading model...");
  network.setup();
  network.setConfidenceThreshold(0.5);

  println("setup camera...");
  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  image(cam, 0, 0);
  
  if(cam.width == 0) {
    return;
  }
  
  detections = network.run(cam);

  noFill();
  strokeWeight(2f);

  stroke(200, 80, 100);
  for (ObjectDetectionResult detection : detections) {
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
  }

  surface.setTitle("Hand Detection Test - FPS: " + Math.round(frameRate));
}
