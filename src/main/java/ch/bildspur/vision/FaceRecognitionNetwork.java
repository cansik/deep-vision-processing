package ch.bildspur.vision;

import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_text.FloatVector;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.transpose;
import static org.bytedeco.opencv.global.opencv_dnn.*;

public class FaceRecognitionNetwork extends DeepNeuralNetwork<List<ObjectDetectionResult>> {
    private Path modelPath;
    protected Net net;

    private int width;
    private int height;

    private float confidenceThreshold = 0.7f;

    private float nmsThreshold = 0.3f; // iou_threshold
    private int topK = -1;

    public FaceRecognitionNetwork(Path modelPath, int width, int height) {
        this.modelPath = modelPath;
        this.width = width;
        this.height = height;
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
    public List<ObjectDetectionResult> run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1 / 127.0,
                new Size(width, height),
                new Scalar(127),
                false, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        // extract boxes and scores
        Mat boxesOut = outs.get(0);
        Mat confidencesOut = outs.get(1);

        Mat boxes = new Mat();
        Mat confidences = new Mat();

        // boxes
        transpose(boxesOut.reshape(1, 4), boxes);

        // class confidences (BACKGROUND, face)
        transpose(confidencesOut.reshape(1, 2), confidences);

        return predict(frame.rows(), frame.cols(), confidences, boxes);
    }

    private List<ObjectDetectionResult> predict(int frameWidth, int frameHeight, Mat confidences, Mat boxes) {
        FloatVector relevantConfidences = new FloatVector();
        RectVector relevantBoxes = new RectVector();

        // todo: implement predict for multi class predictions

        // extract only relevant prob
        for (int i = 0; i < boxes.rows(); i++) {
            FloatPointer confidencesPtr = new FloatPointer(confidences.row(i).data());
            float probability = confidencesPtr.get(1); // read first column (face)

            if (probability < confidenceThreshold) continue;

            // add probability
            relevantConfidences.push_back(probability);

            // add box data
            FloatPointer boxesPtr = new FloatPointer(boxes.row(i).data());
            int left = (int) (boxesPtr.get(0) * frameWidth);
            int top = (int) (boxesPtr.get(1) * frameHeight);
            int width = (int) (boxesPtr.get(2) * frameHeight);
            int height = (int) (boxesPtr.get(3) * frameHeight);
            relevantBoxes.push_back(new Rect(left, top, width, height));
        }

        // run nms
        IntPointer indices = new IntPointer(confidences.size());
        FloatPointer confidencesPointer = new FloatPointer(relevantConfidences.size());
        confidencesPointer.put(relevantConfidences.get());

        NMSBoxes(relevantBoxes, confidencesPointer, confidenceThreshold, nmsThreshold, indices, 1.0f, topK);

        // extract nms result
        List<ObjectDetectionResult> detections = new ArrayList<>();
        for (int i = 0; i < indices.limit(); ++i) {
            int idx = indices.get(i);
            Rect box = relevantBoxes.get(idx);

            detections.add(new ObjectDetectionResult(1, "face", relevantConfidences.get(idx),
                    box.x(), box.y(), box.width(), box.height()));
        }

        return detections;
    }


}
