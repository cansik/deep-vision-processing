package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectSegmentationNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.StringVector;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;

public class MaskRCNN extends ObjectSegmentationNetwork {
    private Path config;
    private Path model;
    private Net net;

    private float maskThreshold = 0.3f;

    public MaskRCNN(Path config, Path model) {
        this.config = config;
        this.model = model;

        this.setConfidenceThreshold(0.5f);
    }

    @Override
    public boolean setup() {
        net = readNetFromTensorflow(
                model.toAbsolutePath().toString(),
                config.toAbsolutePath().toString()
        );

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    public ResultList<ObjectDetectionResult> run(Mat frame) {
        // todo: check if scale / mean / size is correct
        Mat inputBlob = blobFromImage(frame,
                1.0,
                frame.size(),
                new Scalar(0.0),
                true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);


        // todo: implement RCNN
        // help: https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
        return null;
    }

    public float getMaskThreshold() {
        return maskThreshold;
    }

    public void setMaskThreshold(float maskThreshold) {
        this.maskThreshold = maskThreshold;
    }
}
