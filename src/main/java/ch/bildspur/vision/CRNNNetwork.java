package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.TextResult;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTorch;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class CRNNNetwork extends BaseNeuralNetwork<TextResult> {
    private Path model;
    private Net net;

    public CRNNNetwork(Path model) {
        this.model = model;
    }

    @Override
    public boolean setup() {
        net = readNetFromTorch(model.toAbsolutePath().toString());

        if (DeepVision.ENABLE_CUDA_BACKEND) {
            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
        }

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return false;
    }

    @Override
    public TextResult run(Mat frame) {
        // convert to grayscale
        cvtColor(frame, frame, COLOR_RGB2GRAY);

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1 / 255.0,
                new Size(128, 32),
                Scalar.all(0),
                false, true, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        for (int i = 0; i < outNames.size(); i++) {
            System.out.println(outNames.get(i).getString());
        }

        // run detection
        net.forward(outs, outNames);

        return new TextResult("hello", 0.0f);
    }

    public Net getNet() {
        return net;
    }
}
