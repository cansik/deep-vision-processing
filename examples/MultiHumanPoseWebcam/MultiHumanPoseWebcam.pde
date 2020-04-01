import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import java.util.List;

import processing.video.Capture;

Capture cam;

DeepVision vision = new DeepVision(this);

SSDMobileNetwork network;
SingleHumanPoseNetwork pose;

List<ObjectDetectionResult> detections;
List<ObjectDetectionResult> humans = new ArrayList<ObjectDetectionResult>();
List<HumanPoseResult> humanPoseResults = new ArrayList<HumanPoseResult>();

float minProbability = 0.3;

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  println("creating network...");
  network = vision.createMobileNetV2();
  pose = vision.createSingleHumanPoseEstimation();

  print("loading model...");
  network.setup();
  pose.setup();
  println("done!");

  network.setConfidenceThreshold(0.5);

  println("setup camera...");
  String[] cams = Capture.list();
  cam = new Capture(this, cams[1]);
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  image(cam, 0, 0);

  detections = network.run(cam);

  // filter person classes
  humans.clear();
  for (ObjectDetectionResult result : detections) {
    if (result.getClassName().equals("person")) {
      this.humans.add(result);
    }
  }

  humanPoseResults = pose.runByDetections(cam, humans);

  for (int i = 0; i < humans.size(); i++) {
    noFill();
    strokeWeight(2f);

    ObjectDetectionResult detection = humans.get(i);
    HumanPoseResult pose = humanPoseResults.get(i);

    stroke(200, 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

    stroke(300, 80, 100);

    pushMatrix();
    translate(detection.getX(), detection.getY());
    drawHuman(pose);
    popMatrix();
  }

  surface.setTitle("MobileNet Detection Test - FPS: " + Math.round(frameRate));
}

private void drawHuman(HumanPoseResult human) {
  // draw human
  connect(human.getLeftAnkle(), 
    human.getLeftKnee(), 
    human.getLeftHip(), 
    human.getLeftShoulder(), 
    human.getLeftElbow(), 
    human.getLeftWrist());

  connect(human.getRightAnkle(), 
    human.getRightKnee(), 
    human.getRightHip(), 
    human.getRightShoulder(), 
    human.getRightElbow(), 
    human.getRightWrist());

  connect(human.getLeftHip(), human.getRightHip());
  connect(human.getLeftShoulder(), human.getRightShoulder());

  connect(human.getLeftShoulder(), human.getNose(), human.getRightShoulder());
  connect(human.getLeftEar(), human.getLeftEye(), human.getRightEye(), human.getRightEar());
  connect(human.getLeftEye(), human.getNose(), human.getRightEye());

  // draw points
  int i = 0;
  stroke(0, 50, 100);
  for (KeyPointResult point : human.getKeyPoints()) {
    if (point.getProbability() < minProbability) {
      continue;
    }

    fill(0, 0, 0);
    ellipse(point.getX(), point.getY(), 10, 10);

    fill(0, 50, 100);
    textSize(12);
    text(i + " (" + nf(point.getProbability(), 0, 2) + ")", point.getX() + 5, point.getY());
    i++;
  }
}

private void connect(KeyPointResult... keyPoints) {
  for (int i = 0; i < keyPoints.length - 1; i++) {
    KeyPointResult a = keyPoints[i];
    KeyPointResult b = keyPoints[i + 1];

    if (a.getProbability() > minProbability && b.getProbability() > minProbability)
      line(a.getX(), a.getY(), b.getX(), b.getY());
  }
}
