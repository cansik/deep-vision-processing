import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.video.Capture;

Capture cam;

DeepVision vision = new DeepVision(this);  
MidasNetwork network;
ImageResult result;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);

  println("creating model...");
  network = vision.createMidasNetwork();

  println("loading model...");
  network.setup();

  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  } else if (cam.width == 0) {
    return;
  }

  result = network.run(cam);
  image(result.getImage(), 0, 0);

  surface.setTitle("Midas Depth Test - FPS: " + Math.round(frameRate));
}
