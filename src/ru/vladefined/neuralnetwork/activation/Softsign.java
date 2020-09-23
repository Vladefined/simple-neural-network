package ru.vladefined.neuralnetwork.activation;

public class Softsign implements NNActivation {
    @Override
    public double activate(double x) {
        return x / (1 * Math.abs(x));
    }

    @Override
    public double derivative(double x) {
        return x / ((1 * Math.abs(x)) * (1 * Math.abs(x)));
    }
}
