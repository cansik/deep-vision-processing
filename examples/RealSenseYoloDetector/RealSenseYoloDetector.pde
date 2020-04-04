import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.YOLONetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;

import ch.bildspur.realsense.*;

RealSenseCamera camera = new RealSenseCamera(this);
DeepVision vision = new DeepVision(this);

YOLONetwork net;
ResultList<ObjectDetectionResult> result;

void setup()
{
  size(848, 480);
  colorMode(HSB, 360, 100, 100);

  println("creating network...");
  net = vision.createYOLOv3Tiny();

  println("loading model...");
  net.setup();

  net.setConfidenceThreshold(0.1f);
  net.setSkipNonMaximumSuppression(false);

  println("starting camera...");
  if (camera.getDeviceCount() < 1) {
    println("no camera connected!");
    exit();
  }

  camera.enableColorStream(848, 480, 30);
  camera.start();
}

void draw()
{
  background(55);

  // read frames
  camera.readFrames();

  // show color image
  PImage input = camera.getColorImage();
  result = net.run(input);

  image(input, 0, 0);

  noFill();
  strokeWeight(2f);

  for (ObjectDetectionResult detection : result) {
    stroke(round(360.0f * (float) detection.getClassId() / net.getClassNames().size()), 75, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

    textSize(15);
    text(detection.getClassName(), detection.getX(), detection.getY());
  }

  surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
}
