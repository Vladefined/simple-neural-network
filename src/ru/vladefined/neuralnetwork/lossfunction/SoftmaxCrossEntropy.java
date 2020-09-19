package ru.vladefined.neuralnetwork.lossfunction;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.activation.SoftMax;

import java.util.Arrays;

public class SoftmaxCrossEntropy implements LossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double errorSum = 0;
        for (int i = 0; i < output.length; i++) {
            for (double o : output) {
                errorSum += expected[i] * Math.log(((SoftMax) NNActivation.SOFTMAX).activate(o, output));
            }
        }
        return -errorSum;
    }
}
