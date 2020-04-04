package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.nio.file.Files;
import java.nio.file.Path;

public class CascadeClassifierNetwork extends BaseNeuralNetwork<ResultList<ObjectDetectionResult>> {
    private Path model;
    private String className;
    private CascadeClassifier net;

    private double scaleFactor = 1.1;
    private int minNeighbors = 3;
    int flags = 0;

    public CascadeClassifierNetwork(Path model) {
        this(model, "object");
    }

    public CascadeClassifierNetwork(Path model, String className) {
        this.model = model;
        this.className = className;
    }

    @Override
    public boolean setup() {
        if (!Files.exists(model))
            return false;

        net = new CascadeClassifier(model.toAbsolutePath().toString());
        return true;
    }

    @Override
    public ResultList<ObjectDetectionResult> run(Mat frame) {
        ResultList<ObjectDetectionResult> detections = new ResultList<>();
        RectVector detectObjects = new RectVector();

        net.detectMultiScale(frame, detectObjects, scaleFactor, minNeighbors, flags, null, null);

        for (int i = 0; i < detectObjects.size(); i++) {
            Rect detection = detectObjects.get(i);
            detections.add(new ObjectDetectionResult(0, className, 1,
                    detection.x(), detection.y(), detection.width(), detection.height()));
        }

        return detections;
    }

    public double getScaleFactor() {
        return scaleFactor;
    }

    public void setScaleFactor(double scaleFactor) {
        this.scaleFactor = scaleFactor;
    }

    public int getMinNeighbors() {
        return minNeighbors;
    }

    public void setMinNeighbors(int minNeighbors) {
        this.minNeighbors = minNeighbors;
    }

    public int getFlags() {
        return flags;
    }

    public void setFlags(int flags) {
        this.flags = flags;
    }
}
