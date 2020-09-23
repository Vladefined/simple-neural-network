package ru.vladefined.neuralnetwork.lossfunction;

public interface LossFunction {
    LossFunction
            BINARY_CROSS_ENTROPY = new BinaryCrossEntropy(),
            L1 = new L1(),
            L2 = new L2(),
            SOFTMAX_CROSS_ENTROPY = new SoftmaxCrossEntropy();

    default double calculate(double[] output, double[] expected) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            sum += single(output[i], expected[i]);
        }

        return sum / output.length;
    }

    double single(double output, double expected);

    double derivative(double output, double expected);

}
