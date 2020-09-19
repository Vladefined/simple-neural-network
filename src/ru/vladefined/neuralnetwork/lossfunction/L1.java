package ru.vladefined.neuralnetwork.lossfunction;

public class L1 implements LossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double errorSum = 0;
        for (int i = 0; i < output.length; i++) {
            errorSum += Math.abs(output[i] - expected[i]);
        }
        return errorSum;
    }
}
