package ru.vladefined.neuralnetwork.weightinitialization;

public interface WeightInit {
    WeightInit
            XAVIER = new Xavier(),
            RANDOM = new RandomInitialization(),
            ONES = new Ones(),
            ZEROS = new Zeros();

    double initWeight(int in, int out);

}
