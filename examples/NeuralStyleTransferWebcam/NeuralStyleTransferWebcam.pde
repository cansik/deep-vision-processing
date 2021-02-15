import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

import processing.video.Capture;

Capture cam;

DeepVision vision = new DeepVision(this, false);
StyleTransferNetwork network;
ImageResult result;

void setup() {
  size(1280, 428);

  println("creating network...");
  network = vision.createStyleTransfer(Repository.InstanceNormCandy);

  println("loading model...");
  network.setup();

  // start webcam
  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  image(cam, 0, 0);

  if (cam.width == 0) {
    return;
  }
  
  ImageResult result = network.run(cam);

  image(cam, 0, 0);
  image(result.getImage(), 640, 0);
  surface.setTitle("Style Transfer - FPS: " + Math.round(frameRate));
}
