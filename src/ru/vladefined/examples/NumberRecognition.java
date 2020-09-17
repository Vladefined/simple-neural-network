package ru.vladefined.examples;

import ru.vladefined.neuralnetwork.NNUtils;
import ru.vladefined.neuralnetwork.NeuralNetwork;
import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.layers.HiddenLayer;
import ru.vladefined.neuralnetwork.layers.InputLayer;
import ru.vladefined.neuralnetwork.layers.OutputLayer;
import ru.vladefined.neuralnetwork.lossfunction.NNLossFunction;
import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.weightinitialization.NNWeightInitialization;

import java.io.IOException;

public class NumberRecognition {

    public static void main(String[] strings) throws IOException {
        NNDataSet trainDataSet = NNUtils.parseMNISTFromCSV("C:\\mnist_train.csv", 60000);
        NNDataSet testDataSet = NNUtils.parseMNISTFromCSV("C:\\mnist_test.csv", 10000);
        NeuralNetwork network = new NeuralNetwork.Builder()
                .setLearningRate(0.01)
                .useBIAS(true)
                .layers()
                .add(new InputLayer.Builder(trainDataSet.getInputLength()).activation(NNActivation.SOFTSIGN))
                .add(new HiddenLayer.Builder(16).weightInit(NNWeightInitialization.RANDOM).activation(NNActivation.SOFTSIGN))
                .add(new HiddenLayer.Builder(16).weightInit(NNWeightInitialization.RANDOM).activation(NNActivation.SOFTSIGN))
                .add(new OutputLayer.Builder(trainDataSet.getOutputLength()).weightInit(NNWeightInitialization.RANDOM).lossFunction(NNLossFunction.L1).activation(NNActivation.SOFTMAX))
                .compile()
                .build();
        network.setIterationListener(iteration -> {
            if ((iteration + 1) % 100 == 0) {
                System.out.println("============================================================================");
                System.out.println("ITERATION " + (iteration + 1) + "/" + trainDataSet.size());
                System.out.println();
                System.out.println("RESULT - " + NNUtils.vectorToNum(network.feedForward(trainDataSet.get(iteration)[0])));
                System.out.println("EXPECTED - " + NNUtils.vectorToNum(trainDataSet.get(iteration)[1]));
                System.out.println();
                System.out.println("COST - " + network.cost());
                System.out.println("============================================================================");
            }
        });
        network.fit(trainDataSet);
        System.out.println(network.test(testDataSet));
    }

}
