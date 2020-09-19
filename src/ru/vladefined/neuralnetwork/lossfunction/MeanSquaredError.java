package ru.vladefined.neuralnetwork.lossfunction;

public class MeanSquaredError implements LossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double errorSum = 0;
        for (int i = 0; i < output.length; i++) {
            errorSum += (expected[i] - output[i]) * (expected[i] - output[i]);
        }
        return 1.0 / output.length * errorSum;
    }
}
