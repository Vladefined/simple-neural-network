package ru.vladefined.neuralnetwork.weightinitialization;

import java.util.Random;

public class Xavier implements NNWeightInitialization {
    @Override
    public double initWeight(int in, int out) { //Maybe I'm wrong here idk
        double rnd = new Random().nextInt(Math.max(in, out) - Math.min(in, out)) + Math.min(in, out);
        return rnd / Math.max(in, out) * Math.sqrt(1.0 / in);
    }
}