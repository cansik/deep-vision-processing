package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
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
    private Scalar mean = new Scalar(0.485, 0.456, 0.406, 0.0);

    public MidasNetwork(Path model) {
        this.model = model;
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(model.toAbsolutePath().toString());
        return true;
    }

    @Override
    public ImageResult run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1.0,
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
        int height = output.size(2);
        output = output.reshape(1, height);

        // convert to grayscale image
        output.convertTo(output, CV_8UC3);
        //cvResize(output, reshaped, CV_INTER_AREA);

        PImage result = new PImage(output.size().width(), output.size().height());
        CvProcessingUtils.toPImage(output, result);
        return new ImageResult(result);
    }
}
