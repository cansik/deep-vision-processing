package ch.bildspur.vision;

import ch.bildspur.vision.network.DeepNeuralNetwork;
import ch.bildspur.vision.network.DeepNeuralNetworkFactory;
import ch.bildspur.vision.network.YoloNetwork;

public enum NetworkType {
    YOLOv3Tiny(YoloNetwork::new);

    private DeepNeuralNetworkFactory networkFactory;

    NetworkType(DeepNeuralNetworkFactory networkFactory) {
        this.networkFactory = networkFactory;
    }

    protected DeepNeuralNetwork create() {
        return networkFactory.create();
    }
}
