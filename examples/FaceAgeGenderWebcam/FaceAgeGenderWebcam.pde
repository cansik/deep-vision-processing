import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.video.Capture;

Capture cam;

DeepVision vision;
CascadeClassifierNetwork faceNetwork;

GenderNetwork genderNetwork;
AgeNetwork ageNetwork;

ResultList<ObjectDetectionResult> detections;
ResultList<ClassificationResult> genders;
ResultList<ClassificationResult> ages;

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  println("creating network...");
  vision = new DeepVision(this);
  faceNetwork = vision.createCascadeFrontalFace();
  genderNetwork = vision.createGenderClassifier();
  ageNetwork = vision.createAgeClassifier();

  println("loading model...");
  faceNetwork.setup();
  genderNetwork.setup();
  ageNetwork.setup();

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

  genders = genderNetwork.runByDetections(cam, detections);
  ages = ageNetwork.runByDetections(cam, detections);

  noFill();
  strokeWeight(2f);

  stroke(200, 80, 100);
  for (int i = 0; i < detections.size(); i++) {
    ObjectDetectionResult face = detections.get(i);
    ClassificationResult gender = genders.get(i);
    ClassificationResult age = ages.get(i);

    rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
    text(gender.getClassName() + ", " + age.getClassName(), face.getX(), face.getY());
  }

  surface.setTitle("Face Age / Gender Test - FPS: " + Math.round(frameRate));
}
