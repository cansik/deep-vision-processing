import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

PImage testImage;

DeepVision vision = new DeepVision(this);
SuperResolutionNetwork network;
ImageResult result;

void setup() {
  size(1280, 428);

  testImage = loadImage(sketchPath("data/office_small.jpg"));

  println("creating network...");
  network = vision.createSuperResolutionLapSRN(8);

  println("loading model...");
  network.setup();

  println("inferencing...");
  result = network.run(testImage);
  println("done!");

  // test
  testImage.resize(640, 0);

  noLoop();
}

void draw() {
  background(55);
  image(testImage, 0, 0);
  image(result.getImage(), 640, 0, 640, 428);
}
