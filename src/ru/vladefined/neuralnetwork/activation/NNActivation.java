package ru.vladefined.neuralnetwork.activation;

public interface NNActivation {
    NNActivation
            SIGMOID = new Sigmoid(),
            RELU = new RELU(),
            LEAKY_RELU = new LeakyRELU(),
            TANH = new TanH(),
            SOFTMAX = new SoftMax(),
            SOFTSIGN = new SoftSign(),
            SOFTPLUS = new SoftPlus();

    double activate(double x);

    double derivative(double x);
}