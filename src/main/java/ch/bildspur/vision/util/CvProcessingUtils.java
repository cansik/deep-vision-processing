package ch.bildspur.vision.util;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import processing.core.PConstants;
import processing.core.PImage;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import static ch.bildspur.vision.util.MathUtils.clamp;
import static org.bytedeco.opencv.global.opencv_core.merge;
import static org.bytedeco.opencv.global.opencv_core.split;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Adapted from https://github.com/atduskgreg/opencv-processing/blob/master/src/gab/opencv/OpenCV.java
 */
public final class CvProcessingUtils implements PConstants {

    private CvProcessingUtils() {
    }

    public static void ARGBtoBGRA(Mat rgba, Mat bgra) {
        MatVector channels = new MatVector();
        split(rgba, channels);

        MatVector reordered = new MatVector();
        // Starts as ARGB.
        // Make into BGRA.

        reordered.push_back(channels.get(3));
        reordered.push_back(channels.get(2));
        reordered.push_back(channels.get(1));
        reordered.push_back(channels.get(0));

        merge(reordered, bgra);
    }

    /**
     * Convert a 4 channel OpenCV Mat object into
     * pixels to be shoved into a 4 channel ARGB PImage's
     * pixel array.
     *
     * @param m An RGBA Mat we want converted
     * @return An int[] formatted to be the pixels of a PImage
     */
    public static int[] matToARGBPixels(Mat m) {
        int pImageChannels = 4;
        int numPixels = m.cols() * m.rows();
        int[] intPixels = new int[numPixels];
        byte[] matPixels = new byte[numPixels * pImageChannels];

        m.data().get(matPixels);
        ByteBuffer.wrap(matPixels).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(intPixels);
        return intPixels;
    }


    /**
     * Convert an OpenCV Mat object into a PImage
     * to be used in other Processing code.
     * Copies the Mat's pixel data into the PImage's pixel array.
     * Iterates over each pixel in the Mat, i.e. expensive.
     * <p>
     * (Mainly used internally by OpenCV. Inspired by toCv()
     * from KyleMcDonald's ofxCv.)
     *
     * @param m   A Mat you want converted
     * @param img The PImage you want the Mat converted into.
     */
    public static void toPImage(Mat m, PImage img) {
        img.loadPixels();

        if (m.channels() == 3) {
            Mat m2 = new Mat();
            cvtColor(m, m2, COLOR_RGB2RGBA);
            img.pixels = matToARGBPixels(m2);
        } else if (m.channels() == 1) {
            Mat m2 = new Mat();
            cvtColor(m, m2, COLOR_GRAY2RGBA);
            img.pixels = matToARGBPixels(m2);
        } else if (m.channels() == 4) {
            img.pixels = matToARGBPixels(m);
        }

        img.updatePixels();
    }

    /**
     * Convert a Processing PImage to an OpenCV Mat.
     * (Inspired by Kyle McDonald's ofxCv's toOf())
     *
     * @param img The PImage to convert.
     * @param m   The Mat to receive the image data.
     */
    public static void toCv(PImage img, Mat m) {
        // may fixes problem with AWT image in P5 4.0
        int[] matPixels = img.pixels;

        ByteBuffer bb = ByteBuffer.allocate(matPixels.length * 4);
        IntBuffer ib = bb.asIntBuffer();
        ib.put(matPixels);

        byte[] bvals = bb.array();

        m.data().put(bvals);
        ARGBtoBGRA(m, m);
    }

    public static Rect createValidROI(Size imageSize, int x, int y, int width, int height) {
        x = clamp(x, 0, imageSize.width());
        y = clamp(y, 0, imageSize.height());
        width = clamp(x + width, 0, imageSize.width()) - x;
        height = clamp(y + height, 0, imageSize.height()) - y;

        return new Rect(x, y, width, height);
    }
}
