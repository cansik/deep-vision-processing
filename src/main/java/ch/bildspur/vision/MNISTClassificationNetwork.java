package ch.bildspur.vision;

import ch.bildspur.vision.result.ClassificationResult;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class MNISTClassificationNetwork extends ClassificationNetwork<ClassificationResult> {
    private Path modelPath;
    protected Net net;

    private int width = 28;
    private int height = 28;

    public MNISTClassificationNetwork(Path modelPath) {
        this.modelPath = modelPath;
        this.setClassNames("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
    }

    @Override
    public boolean setup() {
        net = readNetFromONNX(modelPath.toAbsolutePath().toString());

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    public ClassificationResult run(Mat frame) {
        // convert to gray
        cvtColor(frame, frame, COLOR_RGB2GRAY);

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1 / 255.0,
                new Size(width, height),
                Scalar.all(0.0),
                false, true, CV_32F);

        // set input
        net.setInput(inputBlob);

        // run detection
        Mat out = net.forward();

        // extract result
        FloatPointer data = new FloatPointer(out.row(0).data());

        // todo: use minmaxidx
        int maxIndex = -1;
        float maxProbability = -1.0f;

        for (int i = 0; i < out.cols(); i++) {
            float probability = data.get(i) / 100f;

            // todo: fix probability issue
            // System.out.println("# " + i + ": " + probability + "%");

            if (probability > maxProbability) {
                maxProbability = probability;
                maxIndex = i;
            }
        }

        return new ClassificationResult(maxIndex, getClassNameOrId(maxIndex), maxProbability);
    }
}
