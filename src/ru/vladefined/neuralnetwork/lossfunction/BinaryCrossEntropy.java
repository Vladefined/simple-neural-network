package ru.vladefined.neuralnetwork.lossfunction;

public class BinaryCrossEntropy implements LossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            sum += single(output[i], expected[i]);
        }
        return -sum;
    }

    @Override
    public double single(double output, double expected) {
        return expected * Math.log(output) + (1 - expected) * Math.log(1 - output);
    }

    @Override
    public double derivative(double output, double expected) {
        return expected - output;
    }
}
