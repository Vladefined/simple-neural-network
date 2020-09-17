package ru.vladefined.neuralnetwork.lossfunction;

public class BinaryCrossEntropy implements NNLossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double errorSum = 0;
        for (int i = 0; i < output.length; i++) {
            errorSum += expected[i] * Math.log(1e-15 + output[i]);
        }
        return -(1.0 / output.length * errorSum);
    }
}
