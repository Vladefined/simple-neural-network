package ru.vladefined.neuralnetwork.activation;

public class Sigmoid implements NNActivation {
    @Override
    public double activate(double x) {
        return Math.sin(x);
    }

    @Override
    public double derivative(double x) {
        return Math.cos(x);
    }

    @Override
    public double weightInitialization() {
        return Math.random() * 2 - 1.0;
    }
}
