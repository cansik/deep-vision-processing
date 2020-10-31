package ch.bildspur.vision.test.tools;

import processing.core.PApplet;

public class StopWatch {
    private long startTime = 0;

    public void start() {
        startTime = System.currentTimeMillis();
    }

    public void stop() {
        long endTime = System.currentTimeMillis();
        long delta = endTime - startTime;

        System.out.printf("Took %dms (%.2fs)\n", delta, delta / 1000.0);
    }
}
