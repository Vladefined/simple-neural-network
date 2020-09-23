package ru.vladefined.neuralnetwork.activation;

import java.util.Arrays;

public class SoftMax implements NNActivation {
    @Override
    public double activate(double x) {
        return activate(x, new double[1]);
    }

    public double activate(double x, double[] neurons) {
        double total = Arrays.stream(neurons).map(Math::exp).sum();
        return Math.exp(x) / total;
    }

    @Override
    public double derivative(double x) {
        return derivative(x, null);
    }

    public double derivative(double x, double[] prevNeurons) {
        return activate(1 - x, prevNeurons);
    }
}
