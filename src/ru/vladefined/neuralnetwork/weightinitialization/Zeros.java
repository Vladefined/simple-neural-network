package ru.vladefined.neuralnetwork.weightinitialization;

public class Zeros implements WeightInit {
    @Override
    public double initWeight(int in, int out) {
        return 0;
    }
}
