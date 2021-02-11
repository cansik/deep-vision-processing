import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

PImage groundTruth;

DeepVision vision = new DeepVision(this);
StyleTransferNetwork network;
ImageResult result;

void setup() {
  size(1280, 428);

  groundTruth = loadImage(sketchPath("data/office.jpg"));

  println("creating network...");
  network = vision.createStyleTransfer(Repository.InstanceNormCandy);

  println("loading model...");
  network.setup();

  println("inferencing...");
  result = network.run(groundTruth);
  println("done!");

  noLoop();
}

public void draw() {
  background(55);
  image(groundTruth, 0, 0);
  image(result.getImage(), 640, 0);
  surface.setTitle("Style Transfer - FPS: " + Math.round(frameRate));
}
