import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import java.util.List;

import processing.video.Capture;

Capture cam;

DeepVision vision;
CascadeClassifierNetwork faceNetwork;
FERPlusEmotionNetwork emotionNetwork;

List<ObjectDetectionResult> detections;
List<ClassificationResult> emotions;

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  println("creating network...");
  vision = new DeepVision(this);
  faceNetwork = vision.createCascadeFrontalFace();
  emotionNetwork = vision.createFERPlusEmotionClassifier();

  println("loading model...");
  faceNetwork.setup();
  emotionNetwork.setup();

  println("setup camera...");
  String[] cams = Capture.list();
  cam = new Capture(this, cams[0]);
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  image(cam, 0, 0);
  detections = faceNetwork.run(cam);

  emotions = emotionNetwork.runByDetections(cam, detections);

  noFill();
  strokeWeight(2f);

  stroke(200, 80, 100);
  for (int i = 0; i < detections.size(); i++) {
    ObjectDetectionResult face = detections.get(i);
    ClassificationResult emotion = emotions.get(i);

    rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
    text(emotion.getClassName(), face.getX(), face.getY());
  }

  surface.setTitle("Face Recognition Test - FPS: " + Math.round(frameRate));
}
