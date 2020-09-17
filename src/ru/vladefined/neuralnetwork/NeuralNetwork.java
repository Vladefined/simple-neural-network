package ru.vladefined.neuralnetwork;

import ru.vladefined.neuralnetwork.activation.*;
import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.modules.NNIterationListener;
import ru.vladefined.neuralnetwork.modules.NNLayers;

public class NeuralNetwork {
    private double learningRate;
    private NNLayers layers;

    private NNIterationListener iterationListener = null;

    private double cost = 0;

    private NeuralNetwork(NNLayers layers, double learningRate) {
        this.layers = layers;
        this.learningRate = learningRate;
    }

    public double[] feedForward(double[] inputs) {
        return layers.feedForward(inputs);
    }

    public double cost() {
        return cost;
    }

    public void fit(double[] input, double[] output) {
        feedForward(input);
        backPropagate(output);
        this.cost = layers.cost(output);
    }

    public void fit(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length) return;
        double cost = 0;
        for (int i = 0; i < inputs.length; i++) {
            feedForward(inputs[i]);
            backPropagate(outputs[i]);
            cost += layers.cost(outputs[i]);
            if (iterationListener != null) iterationListener.onIteration(i);
        }
        this.cost = cost / outputs.length;
    }

    public void fit(NNDataSet dataSet) {
        double cost = 0;
        for (int i = 0; i < dataSet.size(); i++) {
            double[][] io = dataSet.get(i);
            feedForward(io[0]);
            backPropagate(io[1]);
            cost += layers.cost(io[0]);
            if (iterationListener != null) iterationListener.onIteration(i);
        }
        this.cost = cost / dataSet.size();
    }

    public void backPropagate(double[] expected) {
        layers.backPropagate(expected);
    }

    public void setIterationListener(NNIterationListener iterationListener) {
        this.iterationListener = iterationListener;
    }

    public static class Builder {
        private double learningRate = 0.01;
        private NNLayers layers;

        public Builder() {
            layers = new NNLayers(this, learningRate);
        }

        public Builder setLearningRate(double rate) {
            learningRate = rate;

            return this;
        }

        public NNLayers layers() {
            return layers;
        }

        public NeuralNetwork build() {
            return new NeuralNetwork(layers, learningRate);
        }

        public Builder useBIAS(boolean use) {
            layers.setUseBIAS(use);

            return this;
        }

    }
}
