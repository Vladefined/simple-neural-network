package ru.vladefined.neuralnetwork.lossfunction;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.activation.SoftMax;

public class SoftmaxCrossEntropy implements LossFunction {
    @Override
    public double calculate(double[] output, double[] expected) {
        double errorSum = 0;
        for (int i = 0; i < output.length; i++) {
            for (double o : output) {
                errorSum += single(output, o, expected[i]);
            }
        }
        return errorSum;
    }

    public double single(double[] outputs, double output, double expected) {
        return -expected * Math.log(((SoftMax) NNActivation.SOFTMAX).activate(output, outputs));
    }

    @Override
    public double single(double output, double expected) {
        return single(null, 0, 0);
    }

    @Override
    public double derivative(double output, double expected) {
        return output - expected;
    }
}
