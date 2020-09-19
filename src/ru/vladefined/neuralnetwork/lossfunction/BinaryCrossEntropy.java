package ru.vladefined.neuralnetwork.lossfunction;

public class BinaryCrossEntropy implements LossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double errorSum = 0;
        for (int i = 0; i < output.length; i++) {
            errorSum += expected[i] * Math.log(output[i]) + (1 - expected[i]) * Math.log(1 - output[i]);
        }
        return -errorSum;
    }
}
