package ch.bildspur.vision.network;

import ch.bildspur.vision.CvProcessingUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.IntVector;
import processing.core.PImage;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class YoloNetwork extends DeepNeuralNetwork {
    private String configPath = "data/darknet/yolov3-tiny.cfg";
    private String weightsPath = "data/darknet/yolov3-tiny.weights";
    //private String configPath = "data/darknet/yolov3.cfg";
    //private String weightsPath = "data/darknet/yolov3.weights";
    private List<String> names = new ArrayList<>();

    float defaultConfThreshold = 0.5f; // Confidence threshold
    float defaultNMSThreshold = 0.4f;  // Non-maximum suppression threshold

    private Net net;

    public void setup() {
        net = readNetFromDarknet(configPath, weightsPath);

        try {
            names = Files.readAllLines(Paths.get("data/darknet/coco.names"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (net.empty()) {
            System.out.println("Can't load network!");
        }
    }

    public List<YoloDetection> detect(PImage image) {
        return detect(image, defaultConfThreshold, defaultNMSThreshold);
    }

    public List<YoloDetection> detect(PImage image, float confThreshold) {
        return detect(image, confThreshold, defaultNMSThreshold);
    }

    public List<YoloDetection> detect(PImage image, float confThreshold, float nmsThreshold) {
        // read frame and prepare
        Mat frame = new Mat(image.height, image.width, CV_8UC4);
        CvProcessingUtils.toCv(image, frame);
        cvtColor(frame, frame, COLOR_RGBA2RGB);

        //frame = imread("data/dog.jpg");

        // convert image into batch of images
        Mat inputBlob = blobFromImage(frame,
                1 / 255.0,
                new Size(416, 416),
                //new Size(608, 608),
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
        return postprocess(frame, outs, confThreshold, nmsThreshold);
    }

    /**
     * Remove the bounding boxes with low confidence using non-maxima suppression
     * @param frame Input frame
     * @param outs Network outputs
     * @param confThreshold Confidence threshold
     * @param nmsThreshold Non maximum suppression threshold
     * @return
     */
    private List<YoloDetection> postprocess(Mat frame, MatVector outs, float confThreshold, float nmsThreshold)
    {
        IntVector classIds = new IntVector();
        FloatVector confidences = new FloatVector();
        RectVector boxes = new RectVector();

        for (int i = 0; i < outs.size(); ++i)
        {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            Mat result = outs.get(i);
            FloatPointer data = new FloatPointer(result.data());

            for (int j = 0; j < result.rows(); j++)
            {
                Mat scores = result.row(j).colRange(5, result.cols());

                Point classIdPoint = new Point(1);
                DoublePointer confidence = new DoublePointer(1);

                // Get the value and location of the maximum score
                minMaxLoc(scores, null, confidence, null, classIdPoint, null);
                if (confidence.get() > confThreshold)
                {
                    // todo: maybe round instead of floor
                    int centerX = (int)(data.get(0) * frame.cols());
                    int centerY = (int)(data.get(1) * frame.rows());
                    int width = (int)(data.get(2) * frame.cols());
                    int height = (int)(data.get(3) * frame.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x());
                    confidences.push_back((float)confidence.get());
                    boxes.push_back(new Rect(left, top, width, height));
                }
            }
        }

        // skip nms
        if(true) {
            List<YoloDetection> detections = new ArrayList<>();
            for (int i = 0; i < confidences.size(); ++i)
            {
                Rect box = boxes.get(i);

                int classId = classIds.get(i);
                detections.add(new YoloDetection(classId, names.get(classId), confidences.get(i),
                        box.x(), box.y(), box.width(), box.height()));
            }
            return detections;
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        IntPointer indices = new IntPointer(confidences.size());
        FloatPointer confidencesPointer = new FloatPointer(confidences.size());
        confidences.put(confidences);

        NMSBoxes(boxes, confidencesPointer, confThreshold, nmsThreshold, indices, 1.f, 0);

        List<YoloDetection> detections = new ArrayList<>();
        for (int i = 0; i < indices.limit(); ++i)
        {
            int idx = indices.get(i);
            Rect box = boxes.get(idx);

            int classId = classIds.get(idx);
            detections.add(new YoloDetection(classId, names.get(classId), confidences.get(idx),
                    box.x(), box.y(), box.width(), box.height()));
        }

        return detections;
    }

    public List<String> getNames() {
        return names;
    }
}
