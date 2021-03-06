package ru.vladefined.neuralnetwork.weightinitialization;

public class RandomInitialization implements WeightInit {
    @Override
    public double initWeight(int in, int out) {
        return Math.random() * 2 - 1.0;
    }
}
