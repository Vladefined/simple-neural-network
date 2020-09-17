package ru.vladefined.neuralnetwork.activation;

public class TanH implements NNActivation {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        return 1 - activate(x) * activate(x);
    }
}
