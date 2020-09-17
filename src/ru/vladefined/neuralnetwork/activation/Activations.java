package ru.vladefined.neuralnetwork.activation;

public class Activations {
    public static final NNActivation
            SIGMOID = new Sigmoid(),
            RELU = new RELU(),
            LEAKY_RELU = new LeakyRELU(),
            TANH = new TanH(),
            SOFTMAX = new SoftMax(),
            SOFTSIGN = new SoftSign(),
            SOFTPLUS = new SoftPlus();
}
