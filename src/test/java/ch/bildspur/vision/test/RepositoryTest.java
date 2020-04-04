package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import processing.core.PApplet;

public class RepositoryTest extends PApplet {

    public static void main(String... args) {
        RepositoryTest sketch = new RepositoryTest();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480);
    }

    public void setup() {
        DeepVision vision = new DeepVision(this);

        println("clearing repository...");
        vision.clearRepository();

        // download data
        vision.createYOLOv3();

        exit();
    }

    public void draw() {
        background(55);
    }
}
