package ch.bildspur.vision;

import ch.bildspur.vision.network.DeepNeuralNetwork;
import org.opencv.core.Core;
import processing.core.PConstants;

public class DeepVision implements PConstants {

    public DeepVision() {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public DeepNeuralNetwork createNetwork(NetworkType networkType) {
        return networkType.create();
    }
}
