package ru.vladefined.neuralnetwork.weightinitialization;

public interface NNWeightInitialization {
    NNWeightInitialization
            XAVIER = new Xavier(),
            RANDOM = new RandomInitialization();

    double initWeight(int in, int out);

}
