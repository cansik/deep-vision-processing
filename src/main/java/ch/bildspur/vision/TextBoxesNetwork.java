package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectDetectionNetwork;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.TextDetectorCNN;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_dnn.NMSBoxes;

public class TextBoxesNetwork extends ObjectDetectionNetwork {
    private Path protoTextPath;
    private Path modelPath;

    private TextDetectorCNN net;
    private float nmsThreshold = 0.4f;

    public TextBoxesNetwork(Path protoTextPath, Path modelPath) {
        this.protoTextPath = protoTextPath;
        this.modelPath = modelPath;
        this.setConfidenceThreshold(0.3f);

        setLabels("text");
    }

    @Override
    public boolean setup() {
        net = TextDetectorCNN.create(
                protoTextPath.toAbsolutePath().toString(),
                modelPath.toAbsolutePath().toString());

        return true;
    }

    @Override
    public ResultList<ObjectDetectionResult> run(Mat frame) {
        // detect boxes
        RectVector boxes = new RectVector();
        FloatVector confidences = new FloatVector();
        net.detect(frame, boxes, confidences);

        // clean up boxes with nms
        IntPointer indices = new IntPointer(confidences.size());
        FloatPointer confidencesPointer = new FloatPointer(confidences.size());
        confidencesPointer.put(confidences.get());

        NMSBoxes(boxes, confidencesPointer, getConfidenceThreshold(), nmsThreshold, indices);

        // extract relevant results
        ResultList<ObjectDetectionResult> detections = new ResultList<>();
        for (int i = 0; i < indices.limit(); ++i) {
            int idx = indices.get(i);
            Rect box = boxes.get(idx);

            detections.add(new ObjectDetectionResult(0, getLabelOrId(0), confidences.get(idx),
                    box.x(), box.y(), box.width(), box.height()));
        }

        return detections;
    }

    public float getNmsThreshold() {
        return nmsThreshold;
    }

    public void setNmsThreshold(float nmsThreshold) {
        this.nmsThreshold = nmsThreshold;
    }
}
