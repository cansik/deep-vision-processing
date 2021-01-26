import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.SingleHumanPoseNetwork;
import ch.bildspur.vision.result.*;

PImage testImage;

DeepVision vision;

SingleHumanPoseNetwork pose;
HumanPoseResult result;

PImage image;

void setup() {
  size(480, 640, FX2D);

  colorMode(HSB, 360, 100, 100);
  testImage = loadImage(sketchPath("data/pose.jpg"));

  println("creating network...");
  vision = new DeepVision(this);
  pose = vision.createSingleHumanPoseEstimation();

  println("loading model...");
  pose.setup();

  println("inferencing...");
  result = pose.run(testImage);
  println("done!");
}

void draw() {
  background(55);

  image(testImage, 0, 0);

  // draw result
  stroke(180, 80, 100);
  noFill();
  drawHuman(result);

  noFill();
  strokeWeight(2f);

  surface.setTitle("Pose Estimation Test - FPS: " + Math.round(frameRate));
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
  fill(0);
  for (KeyPointResult point : human.getKeyPoints()) {
    ellipse(point.getX(), point.getY(), 10, 10);
    text(i, point.getX() + 5, point.getY());
    i++;
  }
}

private void connect(KeyPointResult... keyPoints) {
  for (int i = 0; i < keyPoints.length - 1; i++) {
    KeyPointResult a = keyPoints[i];
    KeyPointResult b = keyPoints[i + 1];

    line(a.getX(), a.getY(), b.getX(), b.getY());
  }
}
