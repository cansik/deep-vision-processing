package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.IntVector;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_dnn.*;

public class YOLONetwork extends ObjectDetectionNetwork {

    private Path configPath;
    private Path weightsPath;
    private int width;
    private int height;

    private float confidenceThreshold = 0.5f;
    private float nonMaximumSuppressionThreshold = 0.4f;
    private boolean skipNonMaximumSuppression = false;

    private Net net;

    public YOLONetwork(Path configPath, Path weightsPath, int width, int height) {
        this.configPath = configPath;
        this.weightsPath = weightsPath;
        this.width = width;
        this.height = height;
    }

    public boolean setup() {
        net = readNetFromDarknet(
                configPath.toAbsolutePath().toString(),
                weightsPath.toAbsolutePath().toString());

        //net.setPreferableBackend(DNN_BACKEND_OPENCV);
        //net.setPreferableTarget(DNN_TARGET_OPENCL);

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    public List<ObjectDetectionResult> run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1 / 255.0,
                new Size(width, height),
                new Scalar(0.0),
                true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        StringVector outNames = net.getUnconnectedOutLayersNames();
        MatVector outs = new MatVector(outNames.size());

        // run detection
        net.forward(outs, outNames);

        // evaluate result
        return postprocess(frame, outs);
    }

    /**
     * Remove the bounding boxes with low confidence using non-maxima suppression
     *
     * @param frame Input frame
     * @param outs  Network outputs
     * @return List of objects
     */
    private List<ObjectDetectionResult> postprocess(Mat frame, MatVector outs) {
        IntVector classIds = new IntVector();
        FloatVector confidences = new FloatVector();
        RectVector boxes = new RectVector();

        for (int i = 0; i < outs.size(); ++i) {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            Mat result = outs.get(i);

            for (int j = 0; j < result.rows(); j++) {
                FloatPointer data = new FloatPointer(result.row(j).data());
                Mat scores = result.row(j).colRange(5, result.cols());

                Point classIdPoint = new Point(1);
                DoublePointer confidence = new DoublePointer(1);

                // Get the value and location of the maximum score
                minMaxLoc(scores, null, confidence, null, classIdPoint, null);
                if (confidence.get() > confidenceThreshold) {
                    // todo: maybe round instead of floor
                    int centerX = (int) (data.get(0) * frame.cols());
                    int centerY = (int) (data.get(1) * frame.rows());
                    int width = (int) (data.get(2) * frame.cols());
                    int height = (int) (data.get(3) * frame.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x());
                    confidences.push_back((float) confidence.get());
                    boxes.push_back(new Rect(left, top, width, height));
                }
            }
        }

        // skip nms
        if (skipNonMaximumSuppression) {
            List<ObjectDetectionResult> detections = new ArrayList<>();
            for (int i = 0; i < confidences.size(); ++i) {
                Rect box = boxes.get(i);

                int classId = classIds.get(i);
                detections.add(new ObjectDetectionResult(classId, getLabelOrId(classId), confidences.get(i),
                        box.x(), box.y(), box.width(), box.height()));
            }
            return detections;
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        IntPointer indices = new IntPointer(confidences.size());
        FloatPointer confidencesPointer = new FloatPointer(confidences.size());
        confidencesPointer.put(confidences.get());

        NMSBoxes(boxes, confidencesPointer, confidenceThreshold, nonMaximumSuppressionThreshold, indices, 1.f, 0);

        List<ObjectDetectionResult> detections = new ArrayList<>();
        for (int i = 0; i < indices.limit(); ++i) {
            int idx = indices.get(i);
            Rect box = boxes.get(idx);

            int classId = classIds.get(idx);
            detections.add(new ObjectDetectionResult(classId, getLabelOrId(classId), confidences.get(idx),
                    box.x(), box.y(), box.width(), box.height()));
        }

        return detections;
    }

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(float confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }

    public float getNonMaximumSuppressionThreshold() {
        return nonMaximumSuppressionThreshold;
    }

    public void setNonMaximumSuppressionThreshold(float nonMaximumSuppressionThreshold) {
        this.nonMaximumSuppressionThreshold = nonMaximumSuppressionThreshold;
    }

    public boolean isSkipNonMaximumSuppression() {
        return skipNonMaximumSuppression;
    }

    public void setSkipNonMaximumSuppression(boolean skipNonMaximumSuppression) {
        this.skipNonMaximumSuppression = skipNonMaximumSuppression;
    }

    public Path getConfigPath() {
        return configPath;
    }

    public Path getWeightsPath() {
        return weightsPath;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Net getNet() {
        return net;
    }
}
