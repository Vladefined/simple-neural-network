package ru.vladefined.neuralnetwork.weightinitialization;

public class RandomInitialization implements NNWeightInitialization {
    @Override
    public double initWeight(int in, int out) {
        return Math.random() * 2 - 1.0;
    }
}
