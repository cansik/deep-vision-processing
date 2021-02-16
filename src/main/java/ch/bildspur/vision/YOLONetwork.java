package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.IntVector;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;

public class YOLONetwork extends ObjectDetectionNetwork {

    private Path configPath;
    private Path weightsPath;
    private int width;
    private int height;

    private float nmsThreshold = 0.4f;
    private boolean skipNMS = false;

    private Net net;
    private StringVector outNames;
    //private MatVector outs;

    public YOLONetwork(Path configPath, Path weightsPath, int width, int height) {
        this.configPath = configPath;
        this.weightsPath = weightsPath;
        this.width = width;
        this.height = height;
        this.setConfidenceThreshold(0.5f);
    }

    public boolean setup() {
        net = readNetFromDarknet(
                configPath.toAbsolutePath().toString(),
                weightsPath.toAbsolutePath().toString());

        // setup output layers
        outNames = net.getUnconnectedOutLayersNames();
        //outs = new MatVector(outNames.size());

        if (DeepVision.ENABLE_CUDA_BACKEND) {
            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
        }

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    public ResultList<ObjectDetectionResult> run(Mat frame) {
        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1 / 255.0,
                new Size(width, height),
                new Scalar(0.0),
                true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // run detection
        MatVector outs = new MatVector(outNames.size());
        net.forward(outs, outNames);

        // evaluate result
        ResultList<ObjectDetectionResult> result = postprocess(frame, outs);

        // cleanup
        outs.releaseReference();
        inputBlob.release();

        return result;
    }

    /**
     * Remove the bounding boxes with low confidence using non-maxima suppression
     *
     * @param frame Input frame
     * @param outs  Network outputs
     * @return List of objects
     */
    private ResultList<ObjectDetectionResult> postprocess(Mat frame, MatVector outs) {
        IntVector classIds = new IntVector();
        FloatVector confidences = new FloatVector();
        RectVector boxes = new RectVector();

        for (int i = 0; i < outs.size(); ++i) {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            Mat result = outs.get(i);
            FloatIndexer data = result.createIndexer();

            for (int j = 0; j < result.rows(); j++) {
                // minMaxLoc implemented in java
                int maxIndex = -1;
                float maxScore = Float.MIN_VALUE;
                for (int k = 5; k < result.cols(); k++) {
                    float score = data.get(j, k);
                    if (score > maxScore) {
                        maxScore = score;
                        maxIndex = k - 5;
                    }
                }

                if (maxScore > getConfidenceThreshold()) {
                    int centerX = (int) (data.get(j, 0) * frame.cols());
                    int centerY = (int) (data.get(j, 1) * frame.rows());
                    int width = (int) (data.get(j, 2) * frame.cols());
                    int height = (int) (data.get(j, 3) * frame.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(maxIndex);
                    confidences.push_back(maxScore);
                    // todo: creating rects is inefficient
                    boxes.push_back(new Rect(left, top, width, height));
                }
            }

            data.release();
            result.release();
        }

        // skip nms
        if (skipNMS) {
            ResultList<ObjectDetectionResult> detections = new ResultList<>();
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

        NMSBoxes(boxes, confidencesPointer, getConfidenceThreshold(), nmsThreshold, indices, 1.f, 0);

        ResultList<ObjectDetectionResult> detections = new ResultList<>();
        for (int i = 0; i < indices.limit(); ++i) {
            int idx = indices.get(i);
            Rect box = boxes.get(idx);

            int classId = classIds.get(idx);
            detections.add(new ObjectDetectionResult(classId, getLabelOrId(classId), confidences.get(idx),
                    box.x(), box.y(), box.width(), box.height()));
        }

        // cleanup
        indices.releaseReference();
        confidencesPointer.releaseReference();
        classIds.releaseReference();
        confidences.releaseReference();
        boxes.releaseReference();

        return detections;
    }

    public float getNmsThreshold() {
        return nmsThreshold;
    }

    public void setNmsThreshold(float nmsThreshold) {
        this.nmsThreshold = nmsThreshold;
    }

    public boolean isSkipNMS() {
        return skipNMS;
    }

    public void setSkipNMS(boolean skipNMS) {
        this.skipNMS = skipNMS;
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
