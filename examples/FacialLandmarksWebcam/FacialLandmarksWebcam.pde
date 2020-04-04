import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.video.Capture;

Capture cam;

DeepVision vision = new DeepVision(this);
CascadeClassifierNetwork faceNetwork;
FacemarkLBFNetwork facemark;

ResultList<ObjectDetectionResult> detections;
ResultList<FacialLandmarkResult> markedFaces;

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  println("creating network...");
  faceNetwork = vision.createCascadeFrontalFace();
  facemark = vision.createFacemarkLBF();


  println("loading model...");
  faceNetwork.setup();
  facemark.setup();

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

  markedFaces = facemark.runByDetections(cam, detections);

  for (int i = 0; i < detections.size(); i++) {
    ObjectDetectionResult face = detections.get(i);
    FacialLandmarkResult landmarks = markedFaces.get(i);

    noFill();
    strokeWeight(2f);
    stroke(200, 80, 100);
    rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());

    noStroke();
    fill(100, 80, 200);
    for (int j = 0; j < landmarks.getKeyPoints().size(); j++) {
      KeyPointResult kp = landmarks.getKeyPoints().get(j);
      ellipse(kp.getX(), kp.getY(), 5, 5);
    }
  }

  surface.setTitle("Face Recognition Test - FPS: " + Math.round(frameRate));
}
