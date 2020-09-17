package ru.vladefined.neuralnetwork.activation;

public class SoftPlus implements NNActivation {
    @Override
    public double activate(double x) {
        return Math.log(1 + Math.exp(x));
    }

    @Override
    public double derivative(double x) {
        return 1.0 / (1 + Math.exp(x));
    }

    @Override
    public double weightInitialization() {
        return Math.random();
    }
}
