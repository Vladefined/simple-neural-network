package ru.vladefined.neuralnetwork;

import ru.vladefined.neuralnetwork.activation.Activations;
import ru.vladefined.neuralnetwork.layers.HiddenLayer;
import ru.vladefined.neuralnetwork.layers.InputLayer;
import ru.vladefined.neuralnetwork.layers.OutputLayer;

import java.util.Arrays;

public class Main {

    public static void main(String[] strings) {
        NeuralNetwork neuralNetwork = new NeuralNetwork.Builder()
                .setLearningRate(0.1)
                .useBIAS(true)
                .layers()
                .add(new InputLayer.Builder(3).activation(Activations.RELU))
                .add(new HiddenLayer.Builder(6).activation(Activations.RELU).dropout(0.5))
                .add(new OutputLayer.Builder(3).activation(Activations.SIGMOID))
                .compile()
                .build();
        for (int i = 0; i < 1000; i++) {
            neuralNetwork.fit(new double[]{0, 0, 0}, new double[]{0.1, 0.5, 0.1});
            System.out.println("ITERATION " + i + ": " + Arrays.toString(neuralNetwork.feedForward(new double[]{0, 0, 0})));
        }
    }

}
