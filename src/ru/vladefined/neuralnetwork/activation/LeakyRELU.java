package ru.vladefined.neuralnetwork.activation;

public class LeakyRELU implements NNActivation {
    @Override
    public double activate(double x) {
        return x <= 0 ? 0.01 * x : x;
    }

    @Override
    public double derivative(double x) {
        return x <= 0 ? 0.01 : 1;
    }
}
