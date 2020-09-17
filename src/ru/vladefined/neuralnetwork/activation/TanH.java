package ru.vladefined.neuralnetwork.activation;

public class TanH implements NNActivation {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        return 1 - x * x;
    }

    @Override
    public double weightInitialization() {
        return Math.random() * 2 - 1.0;
    }
}
