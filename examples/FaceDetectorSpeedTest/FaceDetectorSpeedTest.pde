import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import java.util.List;
import processing.video.*;

Movie movie;

DeepVision vision;
ULFGFaceDetectionNetwork network;
List<ObjectDetectionResult> detections;

public void setup() {
  size(640, 480, FX2D);
  colorMode(HSB, 360, 100, 100);

  vision = new DeepVision(this);

  movie = new Movie(this, "shibuya.mp4");
  movie.loop();
  movie.speed(2.0);

  println("creating network...");
  network = vision.createULFGFaceDetectorRFB320();

  println("loading model...");
  network.setup();

  //network.setConfidenceThreshold(0.2f);
}

void movieEvent(Movie m) {
  m.read();
}

public void draw() {
  background(55);

  image(movie, 0, 0);
  detections = network.run(movie);

  noFill();
  strokeWeight(2f);

  stroke(200, 80, 100);
  for (ObjectDetectionResult detection : detections) {
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
  }

  surface.setTitle("Face Recognition Speed Test - FPS: " + Math.round(frameRate) + " - " + detections.size() + " Faces");
}
