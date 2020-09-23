package ru.vladefined.neuralnetwork.optimization;

import ru.vladefined.neuralnetwork.activation.SoftMax;
import ru.vladefined.neuralnetwork.layers.OutputLayer;
import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.modules.NNIterationListener;
import ru.vladefined.neuralnetwork.modules.NNLayer;
import ru.vladefined.neuralnetwork.modules.NNLayers;

import java.util.List;

public class StochasticGradientDescent implements OptimizationAlgorithm {
    protected NNLayers layers;
    protected NNIterationListener listener = null;
    protected double cost = 0;

    @Override
    public void setListener(NNIterationListener listener) {
        this.listener = listener;
    }

    @Override
    public void fit(NNLayers layers, NNDataSet dataSet) {
        this.layers = layers;
        for (int i = 0; i < dataSet.size(); i++) {
            double[][] io = dataSet.get(i);
            layers.feedForward(io[0], i != 0);
            backPropagate(io[1]);
            this.cost = layers.cost(io[1]);
            if (listener != null) listener.onIteration(i);
        }
    }

    @Override
    public double cost() {
        return cost;
    }

    private void backPropagate(double[] expected) {
        List<NNLayer> layers = this.layers.layers;
        OutputLayer outputLayer = (OutputLayer) layers.get(layers.size() - 1);
        double[][] lastLocalGradients = new double[outputLayer.weights.length][outputLayer.weights[0].length];
        double biasGradient = 0.0;
        for (int i = 0; i < outputLayer.neurons.length; i++) {
            double errorSignal = outputLayer.lossFunction.derivative(outputLayer.neurons[i], expected[i]);
            NNLayer prevLayer = layers.get(layers.size() - 2);
            for (int j = 0; j < outputLayer.weights[i].length; j++) {
                if (prevLayer.dropout == 0 || Math.random() >= prevLayer.dropout) {
                    double localGrad = outputLayer.calculateGradient(errorSignal, outputLayer.neurons[i], prevLayer);
                    double deltaWeight = this.layers.momentum * outputLayer.lastDeltaWeight[i][j] //Previous Δ weight
                            +
                            this.layers.learningRate * localGrad * prevLayer.neurons[j]; //New Δ weight
                    outputLayer.lastDeltaWeight[i][j] = deltaWeight;
                    lastLocalGradients[i][j] = localGrad;
                    outputLayer.weights[i][j] += deltaWeight;
                }
            }
            if (this.layers.useBIAS) {
                double biasGrad = outputLayer.calculateGradient(errorSignal, outputLayer.neurons[i], prevLayer);
                outputLayer.lastDeltaBias = this.layers.momentum * outputLayer.lastDeltaBias //Old Δ bias
                        +
                        this.layers.learningRate * biasGrad; //New Δ bias
                outputLayer.bias += outputLayer.lastDeltaBias;
                biasGradient = biasGrad;
            }
        }
        for (int i = layers.size() - 2; i > 0; i--) {
            NNLayer prevLayer = layers.get(i - 1);
            NNLayer layer = layers.get(i);
            double[] neurons = layer.neurons;
            double[][] weights = layer.weights;
            double[][] localGradients = new double[weights.length][weights[0].length];
            for (int j = 0; j < neurons.length; j++) {
                if (layer.dropout == 0 || Math.random() >= layer.dropout) {
                    double[] weight = weights[j];
                    for (int k = 0; k < weight.length; k++) {
                        if (prevLayer.dropout == 0 || Math.random() >= prevLayer.dropout) {
                            double localGrad = layer.calculateGradient(neurons[j], layers.get(i + 1).weightsGradientsSum(lastLocalGradients, biasGradient));
                            double deltaWeight = this.layers.momentum * layer.lastDeltaWeight[j][k] //Previous Δ weight
                                    +
                                    this.layers.learningRate * localGrad * prevLayer.neurons[k]; //New Δ weight
                            layer.lastDeltaWeight[j][k] = deltaWeight;
                            localGradients[j][k] = localGrad;
                            weight[k] += deltaWeight;
                        }
                    }
                    if (this.layers.useBIAS) {
                        double biasGrad = layer.calculateGradient(neurons[j], layers.get(i + 1).weightsGradientsSum(lastLocalGradients, biasGradient));
                        layer.lastDeltaBias = this.layers.momentum * layer.lastDeltaBias //Old Δ bias
                                +
                                this.layers.learningRate * biasGrad; //New Δ bias
                        layer.bias += layer.lastDeltaBias;
                        biasGradient = biasGrad;
                    }
                }
            }
            lastLocalGradients = localGradients;
        }
    }
}
