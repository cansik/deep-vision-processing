package ch.bildspur.vision.test;


import processing.core.PApplet;

public class Sketch extends PApplet {

    public static void main(String... args) {
        Sketch sketch = new Sketch();
        sketch.runSketch();
    }

    public void settings() {
        size(640, 480, FX2D);
    }

    public void setup() {

    }

    public void draw() {
        background(55);

        surface.setTitle("RealSense Processing - FPS: " + Math.round(frameRate));
    }
}
