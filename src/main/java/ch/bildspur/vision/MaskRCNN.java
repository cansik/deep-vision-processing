package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;

public class MaskRCNN extends BaseNeuralNetwork<ResultList<ObjectDetectionResult>> {
    private Path config;
    private Path model;
    private Net net;

    public MaskRCNN(Path config, Path model) {
        this.config = config;
        this.model = model;
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
        // todo: implement RCNN
        // help: https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
        return null;
    }
}
