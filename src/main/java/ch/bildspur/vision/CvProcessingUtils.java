package ch.bildspur.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PConstants;
import processing.core.PImage;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC4;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGBA2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static processing.core.PApplet.arrayCopy;

public final class CvProcessingUtils implements PConstants {

    private CvProcessingUtils() {}

    // Convert PImage (ARGB) to Mat (CvType = CV_8UC4)
    public static Mat toMatARGB(PImage image) {
        int w = image.width;
        int h = image.height;

        Mat mat = new Mat(h, w, CV_8UC4);
        byte[] data8 = new byte[w*h*4];
        int[] data32 = new int[w*h];
        arrayCopy(image.pixels, data32);

        ByteBuffer bBuf = ByteBuffer.allocate(w*h*4);
        IntBuffer iBuf = bBuf.asIntBuffer();
        iBuf.put(data32);
        bBuf.get(data8);
        mat.data().put(data8);

        return mat;
    }

    public static Mat toMatRGB(PImage image) {
        Mat mat = toMatARGB(image);
        Mat rgbMat = new Mat();
        cvtColor(mat, rgbMat, COLOR_RGBA2RGB);
        mat.release();
        return rgbMat;
    }

    // Convert Mat (CvType=CV_8UC4) to PImage (ARGB)
    /*
    public static PImage toPImage(Mat mat) {
        int w = mat.width();
        int h = mat.height();

        PImage image = createImage(w, h, ARGB);
        byte[] data8 = new byte[w*h*4];
        int[] data32 = new int[w*h];
        mat.get(0, 0, data8);
        ByteBuffer.wrap(data8).asIntBuffer().get(data32);
        arrayCopy(data32, image.pixels);

        return image;
    }
    */
}
