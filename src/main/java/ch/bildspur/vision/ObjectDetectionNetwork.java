package ch.bildspur.vision;

import ch.bildspur.vision.result.ObjectDetectionResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public abstract class ObjectDetectionNetwork extends DeepNeuralNetwork<List<ObjectDetectionResult>> {
    public static final String COCONamesFile = "data/darknet/coco.names";

    private List<String> names = new ArrayList<>();

    @Override
    boolean setup() {

        return false;
    }

    public void loadNames(String namesFile) {
        try {
            names.clear();
            names.addAll(Files.readAllLines(Paths.get(namesFile)));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<String> getNames() {
        return names;
    }
}
