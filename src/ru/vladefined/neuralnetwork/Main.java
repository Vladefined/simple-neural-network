package ru.vladefined.neuralnetwork;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.layers.HiddenLayer;
import ru.vladefined.neuralnetwork.layers.InputLayer;
import ru.vladefined.neuralnetwork.layers.OutputLayer;
import ru.vladefined.neuralnetwork.weightinitialization.NNWeightInitialization;

import java.util.Arrays;

public class Main {

    public static void main(String[] strings) {
        NeuralNetwork neuralNetwork = new NeuralNetwork.Builder()
                .setLearningRate(0.1)
                .useBIAS(true)
                .layers()
                .add(new InputLayer.Builder(3).activation(NNActivation.LEAKY_RELU))
                .add(new HiddenLayer.Builder(36).activation(NNActivation.LEAKY_RELU).weightInit(NNWeightInitialization.XAVIER))
                .add(new OutputLayer.Builder(3).activation(NNActivation.SIGMOID).weightInit(NNWeightInitialization.XAVIER))
                .compile()
                .build();
        for (int i = 0; i < 1000; i++) {
            neuralNetwork.fit(new double[][]{
                    {0, 0, 0},
                    {1, 0, 1},
                    {1, 1, 1}
            }, new double[][]{
                    {1.0, 1.0, 1.0},
                    {1.0, 0, 1.0},
                    {0, 0, 1.0}
            });
            System.out.println("ITERATION " + i + ": " + Arrays.toString(neuralNetwork.feedForward(new double[]{0, 0, 0})));
            System.out.println("               " + Arrays.toString(neuralNetwork.feedForward(new double[]{1, 0, 1})));
            System.out.println("               " + Arrays.toString(neuralNetwork.feedForward(new double[]{1, 1, 1})));
        }
    }

}
