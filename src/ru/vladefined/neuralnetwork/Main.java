package ru.vladefined.neuralnetwork;

import ru.vladefined.neuralnetwork.activation.Activations;

import java.util.Arrays;

public class Main {

    public static void main(String[] strings) {
        NeuralNetwork neuralNetwork = new NeuralNetwork.Builder()
                .setLearningRate(0.01)
                .useBIAS(true)
                .layers()
                .add(3).activation(Activations.RELU).next()
                .add(12).activation(Activations.RELU).next()
                .add(3).activation(Activations.SIGMOID).next()
                .compile()
                .build();
        for (int i = 0; i < 1000; i++) {
            neuralNetwork.fit(new double[]{0, 0, 0}, new double[]{0.1, 0.5, 0.1});
            System.out.println("ITERATION " + i + ": " + Arrays.toString(neuralNetwork.feedForward(new double[]{0, 0, 0})));
        }
    }

}
