package ru.vladefined.neuralnetwork.lossfunction;

public class L2 implements LossFunction {
    @Override
    public double single(double output, double expected) {
        return (expected - output) * (expected - output);
    }

    @Override
    public double derivative(double output, double expected) {
        return expected - output;
    }
}
