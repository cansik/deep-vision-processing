import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import java.util.List;
import processing.video.*;

Movie movie;

DeepVision vision;
ULFGFaceDetectionNetwork network;
List<ObjectDetectionResult> detections;

ExponentialMovingAverage inferenceTime = new ExponentialMovingAverage(0.05);

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  vision = new DeepVision(this);

  println("creating network...");
  network = vision.createULFGFaceDetectorRFB320();

  println("loading model...");
  network.setup();

  //network.setConfidenceThreshold(0.2f);

  movie = new Movie(this, "shibuya.mp4");
  movie.loop();
  movie.speed(1.0);
  movie.volume(0.0);
}

public void draw() {
  if (movie.available()) {
    movie.read();
  } else {
    return;
  }

  background(55);

  image(movie, 0, 0);

  int start = millis();
  detections = network.run(movie);
  int time = millis() - start;
  float fps = 1000.0 / time;

  noFill();
  strokeWeight(2f);

  stroke(200, 80, 100);
  for (ObjectDetectionResult detection : detections) {
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
  }

  surface.setTitle("Face Recognition Speed Test - Inference FPS: " + nfp(fps, 2, 2) + " - " + detections.size() + " Faces");
}
