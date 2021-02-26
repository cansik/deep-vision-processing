package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.DictValue;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ColorizationNetwork extends BaseNeuralNetwork<ImageResult> {
    private Path protoTextPath;
    private Path modelPath;
    private Path clusterCentersPath;

    private Net net;
    private Scalar mean = new Scalar(-50.0, -50.0, -50.0, 0.0);

    public ColorizationNetwork(Path protoTextPath, Path modelPath, Path clusterCentersPath) {
        this.modelPath = modelPath;
        this.protoTextPath = protoTextPath;
        this.clusterCentersPath = clusterCentersPath;
    }

    @Override
    public boolean setup() {
        net = readNetFromCaffe(protoTextPath.toAbsolutePath().toString(), modelPath.toAbsolutePath().toString());

        // setup network

        // read cluster centers (already transposed and reshaped)
        FileStorage fileStorage = new FileStorage(clusterCentersPath.toAbsolutePath().toString(),
                FileStorage.READ, StandardCharsets.UTF_8.name());
        FileNode clusterCentersNode = fileStorage.get("pts_in_hull_prepared");
        Mat clusterCenters = clusterCentersNode.mat();
        fileStorage.release();

        // prepare input
        MatVector inputBlobs = new MatVector();
        inputBlobs.push_back(clusterCenters);
        net.getLayer(new DictValue("class8_ab")).blobs(inputBlobs);

        // prepare output
        MatVector outputBlobs = new MatVector();
        Mat emptyOutput = new Mat(1, 313, CV_32F);
        emptyOutput.setTo(new Mat(new Scalar(2.606)));
        outputBlobs.push_back(emptyOutput);
        net.getLayer(new DictValue("conv8_313_rh")).blobs(outputBlobs);

        if (DeepVision.ENABLE_CUDA_BACKEND) {
            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
        }

        return true;
    }

    @Override
    public ImageResult run(Mat frame) {
        // create grayscale version
        cvtColor(frame, frame, COLOR_RGB2GRAY);

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1.0 / 255.0,
                new Size(224, 224),
                mean, false, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // run detection
        Mat output = net.forward();

        // post processing
        output.convertTo(output, CV_8U);
        cvtColor(output, output, COLOR_GRAY2BGR);
        resize(output, output, frame.size());

        // convert to processing
        // todo: make that later (keep free of processing)
        PImage result = new PImage(output.size().width(), output.size().height());
        CvProcessingUtils.toPImage(output, result);
        return new ImageResult(result);
    }
}
