package ru.vladefined.neuralnetwork.lossfunction;

public interface NNLossFunction {
    NNLossFunction
            BINARY_CROSS_ENTROPY = new BinaryCrossEntropy(),
            MEAN_SQUARED_ERROR = new MeanSquaredError(),
            L1 = new L1(),
            L2 = new L2();

    double calculate(double[] output, double[] expected);

}
