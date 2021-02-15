import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.util.CvProcessingUtils;

import org.bytedeco.opencv.opencv_core.Mat;

import processing.core.*;
import processing.video.Capture;

import java.util.List;

Capture cam;

DeepVisionPreview vision = new DeepVisionPreview(this);
MaskRCNN network;
List<ObjectSegmentationResult> detections;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);

  println("creating model...");
  network = vision.createMaskRCNN();

  println("loading model...");
  network.setup();

  // setup camera
  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  blendMode(BLEND);
  tint(255, 255);
  image(cam, 0, 0);

  if (cam.width == 0) {
    return;
  }

  detections = network.run(cam);

  // show masks
  noFill();
  strokeWeight(2f);

  for (ObjectSegmentationResult detection : detections) {
    // display rect
    int c = color(round(360.0f * (float) detection.getClassId() / network.getLabels().size()), 75, 100);

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

  surface.setTitle("Mask RCNN - FPS: " + Math.round(frameRate));
}
