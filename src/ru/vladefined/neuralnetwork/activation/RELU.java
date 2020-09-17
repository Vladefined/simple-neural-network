package ru.vladefined.neuralnetwork.activation;

public class RELU implements NNActivation {
    @Override
    public double activate(double x) {
        return x <= 0 ? 0 : x;
    }

    @Override
    public double derivative(double x) {
        return x <= 0 ? 0 : 1;
    }
}
