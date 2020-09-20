package ru.vladefined.neuralnetwork.optimization;

import ru.vladefined.neuralnetwork.activation.SoftMax;
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
            layers.feedForward(io[0]);
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
        double[][] lastLocalGradients = new double[0][];
        double biasGradient = 0.0;
        for (int i = layers.size() - 1; i > 0; i--) {
            NNLayer prevLayer = layers.get(i - 1);
            NNLayer layer = layers.get(i);
            double[] neurons = layer.neurons;
            double[][] weights = layer.weights;
            double[][] localGradients = new double[weights.length][weights[0].length];
            for (int j = 0; j < neurons.length; j++) {
                if (layer.dropout == 0 || Math.random() >= layer.dropout) {
                    double[] weight = weights[j];
                    double errorSignal = i == layers.size() - 1 ? expected[j] - neurons[j] : 0;
                    double NWSum = neuronsWeightsSum(i, j);
                    for (int k = 0; k < weight.length; k++) {
                        if (prevLayer.dropout == 0 || Math.random() >= prevLayer.dropout) {
                            double localGrad;
                            if (i == layers.size() - 1) { //BACK PROP FOR OUTPUT LAYER
                                localGrad = errorSignal * (layer.activation instanceof SoftMax ?
                                        ((SoftMax) layer.activation).derivative(NWSum, prevLayer.neurons, neurons)
                                        :
                                        layer.activation.derivative(NWSum));
                            } else { //BACK PROP FOR HIDDEN LAYER
                                NNLayer nextLayer = layers.get(i + 1);
                                localGrad = layer.activation.derivative(NWSum) * calculateNextLocalGradW(nextLayer, lastLocalGradients, biasGradient);
                            }
                            double deltaWeight = this.layers.momentum * this.layers.learningRate * localGradients[j][k] * prevLayer.neurons[k] //Previous Δ weight
                                    +
                                    this.layers.learningRate * localGrad * prevLayer.neurons[k]; //New Δ weight
                            localGradients[j][k] = localGrad;
                            weight[k] += deltaWeight;
                        }
                    }
                    if (this.layers.useBIAS) {
                        double biasGrad;
                        if (i == layers.size() - 1) { //CORRECT BIAS FOR OUTPUT LAYER
                            biasGrad = errorSignal * (layer.activation instanceof SoftMax ?
                                    ((SoftMax) layer.activation).derivative(NWSum, prevLayer.neurons, neurons)
                                    :
                                    layer.activation.derivative(NWSum));
                        } else { //CORRECT BIAS FOR HIDDEN LAYER
                            NNLayer nextLayer = layers.get(i + 1);
                            biasGrad = layer.activation.derivative(NWSum) * calculateNextLocalGradW(nextLayer, lastLocalGradients, biasGradient);
                        }
                        layer.bias += this.layers.momentum * this.layers.learningRate * biasGradient //Old Δ bias
                                +
                                this.layers.learningRate * biasGrad; //New Δ bias
                        biasGradient = biasGrad;
                    }
                }
            }
            lastLocalGradients = localGradients;
        }
    }

    protected double calculateNextLocalGradW(NNLayer nextLayer, double[][] localGradients, double biasGradient) {
        double nextLocalGradW = 0;
        for (int l = 0; l < nextLayer.weights.length; l++) {
            for (int m = 0; m < nextLayer.weights[l].length; m++) {
                nextLocalGradW += nextLayer.weights[l][m] * localGradients[l][m];
            }
        }
        if (layers.useBIAS) nextLocalGradW += nextLayer.bias * biasGradient;
        return nextLocalGradW;
    }

    protected double neuronsWeightsSum(int layerNum, int neuronNum) {
        NNLayer layer = layers.layers.get(layerNum);
        NNLayer prevLayer = layers.layers.get(layerNum - 1);
        double result = 0;
        for (int i = 0; i < prevLayer.neurons.length; i++) {
            result += prevLayer.neurons[i] * layer.weights[neuronNum][i];
        }
        if (layers.useBIAS) result += prevLayer.bias;

        return result;
    }
}
