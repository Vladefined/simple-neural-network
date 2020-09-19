package ru.vladefined.neuralnetwork.lossfunction;

public interface LossFunction {
    LossFunction
            BINARY_CROSS_ENTROPY = new BinaryCrossEntropy(),
            MEAN_SQUARED_ERROR = new MeanSquaredError(),
            L1 = new L1(),
            L2 = new L2(),
            SQUARED_LOSS = new SquaredLoss(),
            SOFTMAX_CROSS_ENTROPY = new SoftmaxCrossEntropy();

    double calculate(double[] output, double[] expected);

}
