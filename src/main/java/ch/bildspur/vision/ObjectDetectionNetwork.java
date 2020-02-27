package ch.bildspur.vision;

import ch.bildspur.vision.result.ObjectDetectionResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public abstract class ObjectDetectionNetwork extends DeepNeuralNetwork<List<ObjectDetectionResult>> {
    private List<String> names = new ArrayList<>();

    public void loadNames(Path namesFile) {
        try {
            names.clear();
            names.addAll(Files.readAllLines(namesFile.toAbsolutePath()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected String getNameOrId(int classId) {
        if (names.size() > classId) {
            return names.get(classId);
        }

        return String.valueOf(classId);
    }

    public List<String> getNames() {
        return names;
    }
}
