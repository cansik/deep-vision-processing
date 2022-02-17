package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.StringVector;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;

public class StyleTransferNetwork extends BaseNeuralNetwork<ImageResult> {
    private Path model;
    private Net net;
    private Scalar mean = new Scalar(103.939, 116.779, 123.680, 0.0);

    public StyleTransferNetwork(Path model) {
        this.model = model;
    }

    @Override
    public boolean setup() {
        net = readNetFromTorch(model.toAbsolutePath().toString());

        if (DeepVision.ENABLE_CUDA_BACKEND) {
            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
        }

        return true;
    }

    @Override
    public ImageResult run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1.0,
                frame.size(),
                mean, true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        // extract single result
        MatVector images = new MatVector();
        imagesFromBlob(outs.get(0), images);
        Mat output = images.get(0);

        output = add(mean, output).asMat();
        output.convertTo(output, CV_8U);

        // convert to processing
        // todo: make that later (keep free of processing)
        PImage result = new PImage(output.size().width(), output.size().height());
        CvProcessingUtils.toPImage(output, result);
        return new ImageResult(result);
    }

    public Net getNet() {
        return net;
    }
}
