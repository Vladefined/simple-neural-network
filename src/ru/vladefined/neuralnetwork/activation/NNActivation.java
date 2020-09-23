package ru.vladefined.neuralnetwork.activation;

public interface NNActivation {
    NNActivation
            SINUSOID = new Sinusoid(),
            RELU = new RELU(),
            LEAKY_RELU = new LeakyRELU(),
            TANH = new TanH(),
            SOFTMAX = new SoftMax(),
            SOFTSIGN = new Softsign(),
            SIGMOID = new Sigmoid(),
            SOFTPLUS = new SoftPlus(),
            IDENTIFY = new Identify();

    double activate(double x);

    double derivative(double x);
}