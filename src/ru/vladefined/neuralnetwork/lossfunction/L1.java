package ru.vladefined.neuralnetwork.lossfunction;

public class L1 implements LossFunction {
    @Override
    public double single(double output, double expected) {
        return Math.abs(output - expected);
    }

    @Override
    public double derivative(double output, double expected) {
        return output - expected;
    }
}
