package ch.bildspur.vision.util;

public class MathUtils {
    public static float clamp(float val, float min, float max) {
        return Math.max(min, Math.min(max, val));
    }

    public static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    public static int clampByValue(int input, int val, int min, int max) {
        if (val < min)
            return min;

        if (val > max)
            return max;

        return input;
    }
}
