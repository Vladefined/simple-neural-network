package ru.vladefined.examples;

import ru.vladefined.neuralnetwork.modules.NNUtils;
import ru.vladefined.neuralnetwork.NeuralNetwork;
import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.layers.*;
import ru.vladefined.neuralnetwork.lossfunction.LossFunction;
import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.optimization.OptimizationAlgorithm;
import ru.vladefined.neuralnetwork.weightinitialization.WeightInit;

import java.io.IOException;

public class NumberRecognition {

    public static void main(String[] strings) throws IOException {
        NNDataSet trainDataSet = NNUtils.parseMNISTFromCSV("C:\\mnist_train.csv", 60000);
        NNDataSet testDataSet = NNUtils.parseMNISTFromCSV("C:\\mnist_test.csv", 10000);
        NeuralNetwork network = new NeuralNetwork.Builder()
                .learningRate(0.01)
                .momentum(0.5)
                .optimizationAlgorithm(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .layers()
                .add(new InputLayer.Builder(trainDataSet.getInputShape()))
                .add(new HiddenLayer.Builder(32).weightInit(WeightInit.RANDOM).activation(NNActivation.SIGMOID))
                .add(new OutputLayer.Builder(trainDataSet.getOutputShape()).weightInit(WeightInit.RANDOM).lossFunc(LossFunction.L2).activation(NNActivation.SOFTMAX))
                .compile()
                .build();
        network.setIterationListener(iteration -> {
            if ((iteration + 1) % 100 == 0) {
                System.out.println("===========================");
                System.out.println("ITERATION " + (iteration + 1) + "/" + trainDataSet.size());
                System.out.println();
                System.out.println("RESULT - " + NNUtils.vectorToNum(network.feedForward(trainDataSet.get(iteration)[0])));
                System.out.println("EXPECTED - " + NNUtils.vectorToNum(trainDataSet.get(iteration)[1]));
                System.out.println();
                System.out.println("COST - " + network.cost());
                System.out.println("===========================");
            }
        });
        network.fit(trainDataSet);
        System.out.println(network.test(testDataSet));
    }

}
