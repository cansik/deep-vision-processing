package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.network.MultiProcessingNetwork;
import ch.bildspur.vision.network.MultiProcessor;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.TextResult;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.tesseract.TessBaseAPI;
import processing.core.PImage;

import java.nio.file.Path;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class TesseractNetwork extends BaseNeuralNetwork<TextResult> implements MultiProcessingNetwork<TextResult> {
    private Path model;
    private String language;
    private TessBaseAPI api = new TessBaseAPI();
    private final MultiProcessor<TextResult> polyExecutor = new MultiProcessor<>(this);

    public TesseractNetwork(Path model, String language) {
        this.model = model;
        this.language = language;
    }

    @Override
    public boolean setup() {
        // Initialize tesseract-ocr with English, without specifying tessdata path
        if (api.Init(model.toAbsolutePath().getParent().toString(), language, 1) != 0) {
            System.err.println("Could not initialize tesseract.");
            return false;
        }

        return true;
    }

    @Override
    public TextResult run(Mat frame) {
        BytePointer outText;

        Mat gray = new Mat();
        cvtColor(frame, gray, CV_BGR2GRAY);

        api.SetImage(gray.data().asBuffer(),
                gray.size().width(), gray.size().height(),
                gray.channels(), gray.size(1));

        outText = api.GetUTF8Text();
        String text = outText.getString();
        outText.deallocate();

        return new TextResult(text, -1.0f);
    }

    public void release() {
        api.End();
    }

    @Override
    public List<TextResult> runByDetections(PImage image, List<ObjectDetectionResult> detections) {
        return polyExecutor.runByDetections(image, detections);
    }

    @Override
    public List<TextResult> runByDetections(Mat frame, List<ObjectDetectionResult> detections) {
        return polyExecutor.runByDetections(frame, detections);
    }
}
