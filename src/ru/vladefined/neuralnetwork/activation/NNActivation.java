package ru.vladefined.neuralnetwork.activation;

public interface NNActivation {
    double activate(double x);

    double derivative(double x);

    double weightInitialization();
}