package ru.vladefined.examples;

import ru.vladefined.neuralnetwork.NeuralNetwork;
import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.layers.*;
import ru.vladefined.neuralnetwork.lossfunction.LossFunction;
import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.optimization.OptimizationAlgorithm;
import ru.vladefined.neuralnetwork.weightinitialization.WeightInit;

import java.io.File;
import java.io.IOException;

public class SimpleExample {

    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork.Builder()
                .learningRate(0.01)
                .momentum(0.5)
                .optimizationAlgorithm(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .layers()
                .add(new InputLayer.Builder(3))
                .add(new HiddenLayer.Builder(6).weightInit(WeightInit.RANDOM).activation(NNActivation.TANH))
                .add(new OutputLayer.Builder(1).weightInit(WeightInit.RANDOM).lossFunc(LossFunction.BINARY_CROSS_ENTROPY).activation(NNActivation.SINUSOID))
                .compile()
                .build();
        NNDataSet dataSet = new NNDataSet(new double[][] {
                {1.0, 1.0, 1.0},
                {0.0, 1.0, 1.0},
                {1.0, 0.0, 1.0},
                {0.0, 0.0, 1.0}
        }, new double[][] {
                {0.0},
                {0.5},
                {1.0},
                {0.0}
        });
        for (int i = 0; i < 1000; i++) {
            network.fit(dataSet);
            System.out.println("============ EPOCH " + i + " ============");
            System.out.println(network.feedForward(dataSet.get(0)[0])[0]);
            System.out.println(network.feedForward(dataSet.get(1)[0])[0]);
            System.out.println(network.feedForward(dataSet.get(2)[0])[0]);
            System.out.println(network.feedForward(dataSet.get(3)[0])[0]);
            System.out.println();
            System.out.println("COST - " + network.cost());
            System.out.println("===================================");
        }
        try {
            network.save(new File("C:/saved"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
