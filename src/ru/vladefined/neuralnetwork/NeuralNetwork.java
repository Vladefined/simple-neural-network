package ru.vladefined.neuralnetwork;

import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.modules.NNIterationListener;
import ru.vladefined.neuralnetwork.modules.NNLayers;
import ru.vladefined.neuralnetwork.modules.NNUtils;
import ru.vladefined.neuralnetwork.optimization.OptimizationAlgorithm;

public class NeuralNetwork {
    private double learningRate;
    private NNLayers layers;

    private OptimizationAlgorithm algorithm;

    private NeuralNetwork(NNLayers layers, OptimizationAlgorithm algorithm, double learningRate) {
        this.layers = layers;
        this.algorithm = algorithm;
        this.learningRate = learningRate;
    }

    public double[] feedForward(double[] inputs) {
        return layers.feedForward(inputs);
    }

    public double cost() {
        return algorithm.cost();
    }

    public void fit(double[] input, double[] output) {
        algorithm.fit(layers, new NNDataSet(new double[][] {input}, new double[][] {output}));
    }

    public void fit(double[][] inputs, double[][] outputs) {
        algorithm.fit(layers, new NNDataSet(inputs, outputs));
    }

    public double test(NNDataSet dataSet) {
        int right = 0;
        for (int i = 0; i < dataSet.size(); i++) {
            int result = NNUtils.vectorToNum(feedForward(dataSet.get(i)[0]));
            if (result == NNUtils.vectorToNum(dataSet.get(i)[1])) right++;
        }

        return right / (double) dataSet.size();
    }

    public void fit(NNDataSet dataSet) {
        algorithm.fit(layers, dataSet);
    }

    public void setIterationListener(NNIterationListener iterationListener) {
        algorithm.setListener(iterationListener);
    }

    public static class Builder {
        private double learningRate = 0.01, momentum = 0.5;
        private NNLayers layers;
        private OptimizationAlgorithm algorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

        public Builder() {
            layers = new NNLayers(this, learningRate, momentum);
        }

        public Builder learningRate(double rate) {
            learningRate = rate;

            return this;
        }

        public NNLayers layers() {
            return layers;
        }

        public NeuralNetwork build() {
            return new NeuralNetwork(layers, algorithm, learningRate);
        }

        public Builder useBIAS(boolean use) {
            layers.setUseBIAS(use);

            return this;
        }

        public Builder momentum(double momentum) {
            this.momentum = momentum;

            return this;
        }

        public Builder optimizationAlgorithm(OptimizationAlgorithm algorithm) {
            this.algorithm = algorithm;

            return this;
        }

    }
}
