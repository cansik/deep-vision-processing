package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.cvResize;

public class MidasNetwork extends BaseNeuralNetwork<ImageResult> {
    private Path model;
    private Net net;
    // todo: means are from MiDas Net
    private Scalar mean = new Scalar(0.485, 0.456, 0.406, 0.0);

    public MidasNetwork(Path model) {
        this.model = model;
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(model.toAbsolutePath().toString());

        if (DeepVision.ENABLE_CUDA_BACKEND) {
            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
        }

        return true;
    }

    @Override
    public ImageResult run(Mat frame) {
        Size inputSize = frame.size();

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                // todo: is the scale factor correct?!
                1 / 255.0,
                new Size(384, 384),
                mean, true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);
        Mat output = outs.get(0);

        // reshape output mat
        int height = output.size(1);
        int width = output.size(2);

        output = output.reshape(1, height);

        // todo: result a depth frame instead of a color image!
        PImage result = new PImage(output.size().width(), output.size().height());
        mapDepthToImage(output, result);
        result.resize(inputSize.width(), inputSize.height());
        return new ImageResult(result);
    }

    private void mapDepthToImage(Mat depthFrame, PImage img) {
        // find min / max
        DoublePointer minValuePtr = new DoublePointer(1);
        DoublePointer maxValuePtr = new DoublePointer(1);

        minMaxLoc(depthFrame, minValuePtr, maxValuePtr, null, null, null);

        double minValue = minValuePtr.get();
        double maxValue = maxValuePtr.get();

        double distance = maxValue - minValue;
        double minScaled = minValue / distance;

        double alpha = 1.0 / distance * 255.0;
        double beta = -1.0 * minScaled * 255.0;

        depthFrame.convertTo(depthFrame, CV_8U, alpha, beta);
        CvProcessingUtils.toPImage(depthFrame, img);
    }
}
