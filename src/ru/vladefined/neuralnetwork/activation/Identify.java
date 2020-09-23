package ru.vladefined.neuralnetwork.activation;

public class Identify implements NNActivation {
    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }
}
