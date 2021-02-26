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
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
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
        // prepare input frame
        //Mat nlm = new Mat();
        Mat labFrame = new Mat();
        frame.convertTo(labFrame, CV_32FC3, 1 / 255.0, -50.0 / 255.0);
        cvtColor(labFrame, labFrame, COLOR_RGB2Lab);

        // resize
        resize(labFrame, labFrame, new Size(224, 224));

        // extract luminance channel
        MatVector inputChannels = new MatVector();
        split(labFrame, inputChannels);
        Mat luminance = inputChannels.get(0);

        /*
        luminance.convertTo(nlm, CV_8UC1, 255.0, 50.0 / 255.0);
        imshow("NLM", nlm);
        waitKey();
         */

        // convert image into batch of images
        Mat inputBlob = blobFromImage(luminance);

        // set input
        net.setInput(inputBlob);

        // run detection
        Mat out = net.forward();

        // convert 4 dim output [1, 2, 56, 56] int 2 dim [56, 56, 2]
        Mat a = new Mat(56, 56, CV_32F, out.ptr(0, 0));
        Mat b = new Mat(56, 56, CV_32F, out.ptr(0, 1));

        // resize ab and l
        resize(a, a, frame.size());
        resize(b, b, frame.size());
        resize(luminance, luminance, frame.size());

        // merge channels
        Mat output = new Mat(frame.size(), CV_32FC3);
        MatVector outputChannels = new MatVector(0);

        // add predicted a and b
        outputChannels.push_back(luminance);
        outputChannels.push_back(a);
        outputChannels.push_back(b);

        // create new output image
        merge(outputChannels, output);

        // post processing
        cvtColor(output, output, COLOR_Lab2BGR);
        // todo: maybe clip values
        output.convertTo(output, CV_8UC3, 255.0, 50.0 / 255.0);

        // todo: release all mat and matvectors!
        labFrame.release();

        // convert to processing
        // todo: make that later (keep free of processing)
        PImage result = new PImage(output.size().width(), output.size().height());
        CvProcessingUtils.toPImage(output, result);
        return new ImageResult(result);
    }
}
