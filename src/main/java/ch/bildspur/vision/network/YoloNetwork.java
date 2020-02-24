package ch.bildspur.vision.network;

import ch.bildspur.vision.CvProcessingUtils;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;

import processing.core.PImage;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class YoloNetwork extends DeepNeuralNetwork {
    private String configPath = "data/darknet/yolov3-tiny.cfg";
    private String weightsPath = "data/darknet/yolov3-tiny.weights";
    private List<String> names = new ArrayList<>();

    private Net net;

    public void setup() {
        net = readNetFromDarknet(configPath, weightsPath);

        try {
            names = Files.readAllLines(Paths.get("data/darknet/coco.names"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (net.empty()) {
            System.out.println("Can't load network!");
        }
    }

    public void detect(PImage image) {
        Mat cvImage = CvProcessingUtils.toMatRGB(image);

        // convert image into batch of images
        Mat inputBlob = blobFromImage(cvImage, 1 / 255.0, new Size(416, 416), new Scalar(0.0), true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        // evaluate result
        System.out.println(outs);
    }

    public List<String> getNames() {
        return names;
    }
}
