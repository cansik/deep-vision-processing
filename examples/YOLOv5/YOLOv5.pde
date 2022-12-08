import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

DeepVision deepVision = new DeepVision(this);
YOLONetwork yolo;
ResultList<ObjectDetectionResult> detections;

PImage image;
int textSize = 12;

public void setup() {
  size(640, 480);

  colorMode(HSB, 360, 100, 100);

  image = loadImage("pexels-lina-kivaka-5623971.jpg");

  println("creating model...");
  yolo = deepVision.createYOLOv5l();

  println("loading yolo model...");
  yolo.setup();

  println("inferencing...");
  yolo.setConfidenceThreshold(0.95f);
  yolo.setTopK(0);

  detections = yolo.run(image);
}

public void draw() {
  background(55);

  image(image, 0, 0);

  noFill();
  strokeWeight(2f);

  strokeWeight(3f);
  textSize(textSize);

  for (ObjectDetectionResult detection : detections) {
    int hue = (int)(360.0 / yolo.getLabels().size() * detection.getClassId());

    noFill();
    stroke(hue, 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

    fill(hue, 80, 100);
    rect(detection.getX(), detection.getY() - (textSize + 3), textWidth(detection.getClassName()) + 4, textSize + 3);

    fill(0);
    textAlign(LEFT, TOP);
    text(detection.getClassName(), detection.getX() + 2, detection.getY() - textSize - 3);
  }

  surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
}
