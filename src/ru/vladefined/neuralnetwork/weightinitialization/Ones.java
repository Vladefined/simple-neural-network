package ru.vladefined.neuralnetwork.weightinitialization;

public class Ones implements WeightInit {
    @Override
    public double initWeight(int in, int out) {
        return 1;
    }
}
