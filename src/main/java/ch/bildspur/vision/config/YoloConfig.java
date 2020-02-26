package ch.bildspur.vision.config;

import ch.bildspur.vision.ObjectDetectionNetwork;

// todo: maybe replace through factory
public enum YoloConfig {
    YOLOv3_Tiny("data/darknet/yolov3-tiny.cfg",
            "data/darknet/yolov3-tiny.weights",
            ObjectDetectionNetwork.COCONamesFile,
            416, 416),

    YOLOv3_608("data/darknet/yolov3-608.cfg",
            "data/darknet/yolov3.weights",
            ObjectDetectionNetwork.COCONamesFile,
            608, 608);

    private String configPath;
    private String weightsPath;
    private String namesPath;
    private int width;
    private int height;

    YoloConfig(String configPath, String weightsPath, String namesPath, int width, int height) {
        this.configPath = configPath;
        this.weightsPath = weightsPath;
        this.namesPath = namesPath;
        this.width = width;
        this.height = height;
    }

    public String getConfigPath() {
        return configPath;
    }

    public String getWeightsPath() {
        return weightsPath;
    }

    public String getNamesPath() {
        return namesPath;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}
