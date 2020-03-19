package ch.bildspur.vision.network;

import org.bytedeco.opencv.opencv_dnn.Net;

public interface NetworkFactory {
    Net createNetwork();
}
