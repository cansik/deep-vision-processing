package ch.bildspur.vision.test;


import ch.bildspur.vision.DeepVision;
import ch.bildspur.vision.pipeline.ObjectDetectionPipeline;
import ch.bildspur.vision.result.ClassificationResult;
import ch.bildspur.vision.result.DetectionPipelineResult;
import ch.bildspur.vision.result.ObjectDetectionResult;
import ch.bildspur.vision.result.ResultList;
import processing.core.PApplet;
import processing.core.PImage;

public class HumanAttributesPipelineTest extends PApplet {

    public static void main(String... args) {
        HumanAttributesPipelineTest sketch = new HumanAttributesPipelineTest();
        sketch.runSketch();
    }

    public void settings() {
        size(1280, 720);
    }

    PImage testImage;

    DeepVision vision = new DeepVision(this);
    ObjectDetectionPipeline<ClassificationResult> pipeline;
    ResultList<DetectionPipelineResult<ClassificationResult>> results;

    public void setup() {
        colorMode(HSB, 360, 100, 100);

        testImage = loadImage(sketchPath("data/faces.png"));

        println("creating pipeline...");
        pipeline = new ObjectDetectionPipeline<>(
                vision.createCascadeFrontalFace(),
                vision.createGenderClassifier(),
                vision.createAgeClassifier(),
                vision.createFERPlusEmotionClassifier()
        );

        println("loading pipeline...");
        pipeline.setup();

        println("inference pipeline...");
        results = pipeline.run(testImage);

        for (DetectionPipelineResult<ClassificationResult> result : results) {
            ObjectDetectionResult face = result.getDetection();
            ClassificationResult gender = result.getResults().get(0);
            ClassificationResult age = result.getResults().get(1);
            ClassificationResult emotion = result.getResults().get(2);

            System.out.println(face.getClassName() + "\t[" + face.getConfidence() + "] is "
                    + gender.getClassName() + "\t[" + gender.getConfidence() + "] and "
                    + age.getClassName() + "\t[" + age.getConfidence() + "] and "
                    + emotion.getClassName() + "\t[" + emotion.getConfidence() + "]");
        }

        println("found " + results.size() + " faces!");
    }

    public void draw() {
        background(55);

        image(testImage, 0, 0);

        noFill();
        strokeWeight(2f);

        stroke(200, 80, 100);
        for (DetectionPipelineResult<ClassificationResult> result : results) {
            ObjectDetectionResult face = result.getDetection();
            ClassificationResult gender = result.getResults().get(0);
            ClassificationResult age = result.getResults().get(1);
            ClassificationResult emotion = result.getResults().get(2);

            rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
            text(gender.getClassName() + ", " + age.getClassName() + ", " + emotion.getClassName(),
                    face.getX(), face.getY());
        }

        surface.setTitle("Human Attributes Pipeline - FPS: " + Math.round(frameRate));
    }
}
