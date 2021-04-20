import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.video.Capture;

import java.nio.file.*;


// repository
final String maskRepository = "https://github.com/cansik/yolo-mask-detection/releases/download/pre-trained/";

Capture cam;

DeepVision deepVision = new DeepVision(this);
YOLONetwork network;
ResultList<ObjectDetectionResult> detections;

int textSize = 12;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);

  // define models
  final Dependency maskConfig = new Dependency("mask-yolov3-tiny-prn.cfg", maskRepository + "mask-yolov3-tiny-prn.cfg");
  final Dependency maskWeights = new Dependency("mask-yolov3-tiny-prn.weights", maskRepository + "mask-yolov3-tiny-prn.weights");

  // create local networks folder
  Repository.localStorageDirectory = Paths.get(sketchPath("networks"));
  try {
    Files.createDirectories(Repository.localStorageDirectory);
  }
  catch (Exception ex) {
    println("directory already exists...");
  }

  println("downloading models...");
  maskConfig.resolve();
  maskWeights.resolve();

  println("creating model...");
  network = new YOLONetwork(maskConfig.getPath(), maskWeights.getPath(), 416, 416);
  network.setup();
  network.setLabels("good", "bad", "none");

  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  image(cam, 0, 0);

  if (cam.width == 0) {
    return;
  }

  network.setConfidenceThreshold(0.2f);
  detections = network.run(cam);

  strokeWeight(3f);
  textSize(textSize);

  for (ObjectDetectionResult detection : detections) {
    int hue = (int)(360.0 / network.getLabels().size() * detection.getClassId());

    noFill();
    stroke(hue, 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

    fill(hue, 80, 100);
    rect(detection.getX(), detection.getY() - (textSize + 3), textWidth(detection.getClassName()) + 4, textSize + 3);

    fill(0);
    textAlign(LEFT, TOP);
    text(detection.getClassName(), detection.getX() + 2, detection.getY() - textSize - 3);
  }

  surface.setTitle("FaceMask Detector - FPS: " + Math.round(frameRate));
}
