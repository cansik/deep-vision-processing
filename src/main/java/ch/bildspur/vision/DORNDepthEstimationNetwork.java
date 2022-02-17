package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;

public class DORNDepthEstimationNetwork extends BaseNeuralNetwork<ImageResult> {
    private Path protoTextPath;
    private Path modelPath;
    private Net net;
    private Scalar mean = new Scalar(103.0626, 115.9029, 123.1516, 0.0);

    public DORNDepthEstimationNetwork(Path protoTextPath, Path modelPath) {
        this.protoTextPath = protoTextPath;
        this.modelPath = modelPath;
    }

    @Override
    public boolean setup() {
        net = readNetFromCaffe(protoTextPath.toAbsolutePath().toString(), modelPath.toAbsolutePath().toString());

        DeepVision.enableDesiredBackend(net);

        return true;
    }

    @Override
    public ImageResult run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1.0,
                new Size(353, 257),
                mean, true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // print outputs
        for (int i = 0; i < outNames.size(); i++)
            System.out.println(outNames.get(i).getString());

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
