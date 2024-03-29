package ch.bildspur.vision;

import ch.bildspur.vision.network.ObjectSegmentationNetwork;
import ch.bildspur.vision.result.ObjectSegmentationResult;
import ch.bildspur.vision.result.ResultList;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromTensorflow;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class MaskRCNN extends ObjectSegmentationNetwork {
    private Path configPath;
    private Path modelPath;
    private Path labelsPath;
    private Net net;

    private StringVector outNames;

    private float maskThreshold = 0.3f;

    public MaskRCNN(Path configPath, Path modelPath, Path labelsPath) {
        this.configPath = configPath;
        this.modelPath = modelPath;
        this.labelsPath = labelsPath;

        this.setConfidenceThreshold(0.5f);
    }

    @Override
    public boolean setup() {
        net = readNetFromTensorflow(
                modelPath.toAbsolutePath().toString(),
                configPath.toAbsolutePath().toString()
        );

        outNames = new StringVector();
        outNames.push_back("detection_out_final");
        outNames.push_back("detection_masks");

        DeepVision.enableDesiredBackend(net);

        this.loadLabels(labelsPath);

        if (net.empty()) {
            System.out.println("Can't load network!");
            return false;
        }

        return true;
    }

    @Override
    public ResultList<ObjectSegmentationResult> run(Mat frame) {
        // todo: check if scale / mean / size is correct
        Mat inputBlob = blobFromImage(frame,
                1.0,
                frame.size(),
                new Scalar(0.0),
                true, false, CV_32F);

        // set input
        net.setInput(inputBlob);

        // create output layers
        MatVector outs = new MatVector(outNames.size());


        // run detection
        net.forward(outs, outNames);

        Mat boxes = outs.get(0);
        Mat masks = outs.get(1);

        // post processing
        ResultList<ObjectSegmentationResult> result = postProcess(boxes, masks, frame.size());

        // cleanup
        boxes.release();
        masks.release();
        inputBlob.release();
        outs.releaseReference();

        return result;
    }

    private ResultList<ObjectSegmentationResult> postProcess(Mat boxes, Mat masks, Size size) {
        ResultList<ObjectSegmentationResult> results = new ResultList<>();

        int numClasses = masks.size(1);
        int numDetections = boxes.size(2);

        // reshape 4d output
        boxes = boxes.reshape(1, (int) boxes.total() / 7);
        FloatIndexer data = boxes.createIndexer();

        for (int i = 0; i < numDetections; i++) {
            float score = data.get(i, 2);

            // skip if not relevant
            if (score < getConfidenceThreshold())
                continue;

            // extract information
            int classId = (int) data.get(i, 1);
            int left = Math.round(data.get(i, 3) * size.width());
            int top = Math.round(data.get(i, 4) * size.height());
            int right = Math.round(data.get(i, 5) * size.width());
            int bottom = Math.round(data.get(i, 6) * size.height());

            int width = right - left;
            int height = bottom - top;

            // extracting mask, resize and convert to 8bit channel 1
            Mat objectMask = new Mat(masks.size(2), masks.size(3), CV_32F, masks.ptr(i, classId));
            resize(objectMask, objectMask, new Size(width, height));
            objectMask.convertTo(objectMask, CV_8UC1, 255.0, 0.0);

            results.add(new ObjectSegmentationResult(classId, getLabelOrId(classId), score, left, top, width, height, objectMask));
        }

        return results;
    }

    public float getMaskThreshold() {
        return maskThreshold;
    }

    public void setMaskThreshold(float maskThreshold) {
        this.maskThreshold = maskThreshold;
    }

    public Net getNet() {
        return net;
    }
}
