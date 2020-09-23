package ru.vladefined.neuralnetwork.modules.lstm;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.modules.DoubleMatrix;

public class LSTMCell {
    //https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
    private double[]
            inputGate1, inputGate2, forgetGate, outputGate,
            states, outputs,
            gatesWeights, gatesBiases;
    private DoubleMatrix inputWeights, lastInputs = null;
    private double learningRate;
    private LSTMLayer layer;

    public LSTMCell(int inputs, LSTMLayer layer) {
        this.layer = layer;
        this.learningRate = layer.learningRate;
        inputWeights = DoubleMatrix.random(-1, 1, inputs, 4);
        gatesWeights = DoubleMatrix.random(-1, 1, 4, 1).getRow(0);
        gatesBiases = DoubleMatrix.random(-1, 1, 4, 1).getRow(0);
    }

    protected void prepare(int length) {
        states = new double[length];
        outputs = new double[length];
        inputGate1 = new double[length];
        inputGate2 = new double[length];
        forgetGate = new double[length];
        outputGate = new double[length];
    }

    protected double forward(double[] inputs) {
        double lastState = states[0];
        double lastOutput = outputs[0];
        DoubleMatrix dotIW = inputWeights.dot(inputs);
        double tempInputGate1 = NNActivation.TANH.activate(dotIW.get(0, 0) + gatesWeights[0] * lastOutput + gatesBiases[0]);
        double tempInputGate2 = NNActivation.SIGMOID.activate(dotIW.get(0, 1) + gatesWeights[1] * lastOutput + gatesBiases[1]);
        double inputGate = tempInputGate1 * tempInputGate2;
        double forgetGate = NNActivation.SIGMOID.activate(dotIW.get(0, 2) + gatesWeights[2] * lastOutput + gatesBiases[2]);
        double outputGate = NNActivation.SIGMOID.activate(dotIW.get(0, 3) + gatesWeights[3] * lastOutput + gatesBiases[3]);
        states[0] = inputGate + forgetGate * lastState;
        outputs[0] = NNActivation.TANH.activate(states[0]) * outputGate;
        return outputs[0];
    }

    protected void forwardLearning(DoubleMatrix inputs) {
        lastInputs = inputs;
        prepare(inputs.getH());
        for (int i = 0; i < inputs.getH(); i++) {
            double lastState = i > 0 ? states[i - 1] : 0;
            double lastOutput = i > 0 ? outputs[i - 1] : 0;
            DoubleMatrix dotIW = inputWeights.dot(inputs.getRow(i));
            inputGate1[i] = NNActivation.TANH.activate(dotIW.get(0, 0) + gatesWeights[0] * lastOutput + gatesBiases[0]);
            inputGate2[i] = NNActivation.SIGMOID.activate(dotIW.get(0, 1) + gatesWeights[1] * lastOutput + gatesBiases[1]);
            double inputGate = inputGate1[i] * inputGate2[i];
            forgetGate[i] = NNActivation.SIGMOID.activate(dotIW.get(0, 2) + gatesWeights[2] * lastOutput + gatesBiases[2]);
            outputGate[i] = NNActivation.SIGMOID.activate(dotIW.get(0, 3) + gatesWeights[3] * lastOutput + gatesBiases[3]);
            states[i] = inputGate + forgetGate[i] * lastState;
            outputs[i] = NNActivation.TANH.activate(states[i]) * outputGate[i];
        }
    }

    protected double backwardLearning(double[] expected) {
        double deltaOutput = 0;
        double stateGradient = 0;
        double cost = 0;
        double[][] inputWeights = this.inputWeights.getMatrix();
        for (int i = outputs.length - 1; i > -1; i--) {
            cost += layer.lossFunction.single(expected[i], outputs[i]);
            double deltaLoss = layer.lossFunction.derivative(expected[i], outputs[i]);
            double outputGradient = deltaLoss + deltaOutput;
            stateGradient = outputGradient * outputGate[i] * (1 - NNActivation.TANH.activate(states[i]) * NNActivation.TANH.activate(states[i])) + stateGradient * (i != outputs.length - 1 ? forgetGate[i + 1] : 0);
            double inputGate1Gradient = stateGradient * inputGate2[i] * (1 - inputGate1[i] * inputGate1[i]);
            double inputGate2Gradient = stateGradient * inputGate1[i] * inputGate2[i] * (1 - inputGate2[i]);
            double forgetGateGradient = stateGradient * (i != 0 ? states[i - 1] : 0) * forgetGate[i] * (1 - forgetGate[i]);
            double outputGateGradient = outputGradient * NNActivation.TANH.activate(states[i]) * outputGate[i] * (1 - outputGate[i]);
            double[] gatesGradients = new double[] {inputGate1Gradient, inputGate2Gradient, forgetGateGradient, outputGateGradient};
            deltaOutput = dot(gatesWeights, gatesGradients);

            //Sum input gradients
            double[] inputs = lastInputs.getRow(i);
            for (int y = 0; y < gatesGradients.length; y++) {
                for (int x = 0; x < inputs.length; x++) {
                    inputWeights[y][x] -= learningRate * gatesGradients[y] * inputs[x];
                }
            }

            //Sum hidden gradients and biases gradients
            for (int y = 0; y < gatesGradients.length; y++) {
                if (i != 0) gatesWeights[y] -= learningRate * gatesGradients[y] * outputs[i];
                gatesBiases[y] -= learningRate * gatesGradients[y];
            }
        }
        return cost / outputs.length;
    }

    private double dot(double[] vec1, double[] vec2) {
        double sum = 0;
        for (int i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    private double lossDerivative(double output, double expected) {
        return output - expected;
    }

}
