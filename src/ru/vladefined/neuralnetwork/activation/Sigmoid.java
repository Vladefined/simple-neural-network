package ru.vladefined.neuralnetwork.activation;

public class Sigmoid implements NNActivation {
    @Override
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        return x * (1 - x);
    }
}
