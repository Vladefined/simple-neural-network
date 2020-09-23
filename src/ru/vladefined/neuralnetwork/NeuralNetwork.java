package ru.vladefined.neuralnetwork;

import ru.vladefined.neuralnetwork.modules.*;
import ru.vladefined.neuralnetwork.optimization.OptimizationAlgorithm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    private NNLayers layers;

    private OptimizationAlgorithm algorithm;

    private NeuralNetwork(NNLayers layers, OptimizationAlgorithm algorithm) {
        this.layers = layers;
        this.algorithm = algorithm;
    }

    public double[] feedForward(double[] inputs) {
        return feedForward(inputs, false);
    }

    public double[] feedForward(double[] inputs, boolean isLearning) {
        return layers.feedForward(inputs, isLearning);
    }

    public double cost() {
        return algorithm.cost();
    }

    public void fit(double[] input, double[] output) {
        algorithm.fit(layers, new NNDataSet(new double[][]{input}, new double[][]{output}));
    }

    public void fit(double[][] inputs, double[][] outputs) {
        algorithm.fit(layers, new NNDataSet(inputs, outputs));
    }

    public double test(NNDataSet dataSet) {
        int right = 0;
        for (int i = 0; i < dataSet.size(); i++) {
            int result = NNUtils.vectorToNum(feedForward(dataSet.get(i)[0], false));
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

    public void save(File path) throws IOException {
        StringBuilder data = new StringBuilder();
        List<NNLayer> layers = this.layers.layers;
        data.append(layers.size()).append(",");
        for (NNLayer layer : layers) {
            data.append(layer.neurons.length).append(",");
        }
        for (int l = 1; l < layers.size(); l++) {
            NNLayer layer = layers.get(l);
            for (int i = 0; i < layer.weights.length; i++) {
                for (int j = 0; j < layer.weights[i].length; j++) {
                    data.append(layer.weights[i][j]).append(",");
                }
            }
        }
        Files.writeString(path.toPath(), data.substring(0, data.length() - 1));
    }

    public void load(File path) throws IOException {
        String data = Files.readString(path.toPath());
        String[] values = data.split(",");
        int length = Integer.parseInt(values[0]);
        if (layers.layers.size() != length) throw new RuntimeException("Wrong network architecture! Expected " + length + " layers");
        int[] shape = new int[length];
        for (int i = 0; i < length; i++) {
            shape[i] = Integer.parseInt(values[i + 1]);
        }
        for (int i = 0; i < layers.layers.size(); i++) {
            if (layers.layers.get(i).neurons.length != shape[i]) throw new RuntimeException("Wrong network architecture! Expected shape: " + Arrays.toString(shape));
        }
        int layer = 1, weightPos = 0, neuron = 0;
        for (int i = length + 1; i < values.length; i++) {
            layers.layers.get(layer).weights[neuron][weightPos++] = Double.parseDouble(values[i]);
            if (weightPos >= shape[layer - 1]) {
                weightPos = 0;
                neuron++;
            }
            if (neuron >= shape[layer]) {
                layer++;
                neuron = 0;
            }
        }
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
            return new NeuralNetwork(layers, algorithm);
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
