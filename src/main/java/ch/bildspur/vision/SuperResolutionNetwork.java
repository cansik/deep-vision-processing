package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.global.opencv_quality;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_dnn_superres.DnnSuperResImpl;
import processing.core.PImage;

import java.nio.file.Path;

public class SuperResolutionNetwork extends BaseNeuralNetwork<ImageResult> {
    private Path model;
    private String name;
    private int scale;
    private DnnSuperResImpl net;

    public SuperResolutionNetwork(Path model, String name, int scale) {
        this.model = model;
        this.name = name;
        this.scale = scale;

        // fix: https://github.com/bytedeco/javacv/issues/1396
        Loader.load(opencv_quality.class);
    }

    @Override
    public boolean setup() {
        net = new DnnSuperResImpl();
        net.readModel(model.toAbsolutePath().toString());
        net.setModel(name, scale);

        net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
        net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);

        return true;
    }

    @Override
    public ImageResult run(Mat frame) {
        Mat highResImage = new Mat();
        net.upsample(frame, highResImage);

        PImage result = new PImage(highResImage.size().width(), highResImage.size().height());
        CvProcessingUtils.toPImage(highResImage, result);
        return new ImageResult(result);
    }

}
