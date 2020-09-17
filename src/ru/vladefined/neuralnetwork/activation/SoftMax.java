package ru.vladefined.neuralnetwork.activation;

import java.util.Arrays;

public class SoftMax implements NNActivation {
    @Override
    public double activate(double x) {
        return activate(x, null);
    }

    public double activate(double x, double[] neurons) {
        double total = Arrays.stream(neurons).map(Math::exp).sum();
        return Math.exp(x) / total;
    }

    @Override
    public double derivative(double x) {
        return derivative(x, null, null);
    }

    public double derivative(double x, double[] prevNeurons, double[] neurons) {
        return activate(1 - activate(x, neurons), prevNeurons);
    }

    @Override
    public double weightInitialization() {
        return Math.random();
    }
}
