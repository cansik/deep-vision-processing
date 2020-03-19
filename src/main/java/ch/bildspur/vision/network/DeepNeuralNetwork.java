package ch.bildspur.vision.network;

import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import processing.core.PImage;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC4;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGBA2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public abstract class DeepNeuralNetwork<R> {

    public abstract boolean setup();

    public R run(PImage image) {
        // prepare frame
        Mat frame = new Mat(image.height, image.width, CV_8UC4);
        CvProcessingUtils.toCv(image, frame);
        cvtColor(frame, frame, COLOR_RGBA2RGB);

        // inference
        return run(frame);
    }

    public abstract R run(Mat frame);
}
