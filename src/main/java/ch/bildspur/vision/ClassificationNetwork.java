package ch.bildspur.vision;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class ClassificationNetwork<T> extends DeepNeuralNetwork<T> {
    private List<String> classNames = new ArrayList<>();

    public void loadClassNames(Path namesFile) {
        try {
            classNames.clear();
            classNames.addAll(Files.readAllLines(namesFile.toAbsolutePath()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setClassNames(String... names) {
        classNames.clear();
        Collections.addAll(classNames, names);
    }

    protected String getClassNameOrId(int classId) {
        if (classNames.size() > classId) {
            return classNames.get(classId);
        }

        return String.valueOf(classId);
    }

    public List<String> getClassNames() {
        return classNames;
    }
}
