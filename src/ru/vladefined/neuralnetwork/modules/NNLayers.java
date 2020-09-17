package ru.vladefined.neuralnetwork.modules;

import ru.vladefined.neuralnetwork.NeuralNetwork;
import ru.vladefined.neuralnetwork.activation.SoftMax;
import ru.vladefined.neuralnetwork.layers.HiddenLayer;
import ru.vladefined.neuralnetwork.layers.InputLayer;
import ru.vladefined.neuralnetwork.layers.OutputLayer;

import java.util.ArrayList;
import java.util.List;

public class NNLayers {
    private NeuralNetwork.Builder builder;
    protected List<NNLayer> layers = new ArrayList<>();
    private double learningRate;
    protected boolean useBIAS = false;

    public NNLayers(NeuralNetwork.Builder builder, double learningRate) {
        this.builder = builder;
        this.learningRate = learningRate;
    }

    public NeuralNetwork.Builder compile() {
        if (layers.size() < 2) throw new RuntimeException("Neural Network must contain at least 2 layers!");
        if (!(layers.get(0) instanceof InputLayer)) throw new RuntimeException("First layer must be InputLayer!");
        if (!(layers.get(layers.size() - 1) instanceof OutputLayer)) throw new RuntimeException("Last layer must be OutputLayer!");
        for (int i = 1; i < layers.size() - 1; i++) {
            if (!(layers.get(i) instanceof HiddenLayer)) throw new RuntimeException("Layer at index " + i + " must be HiddenLayer!");
        }
        return builder;
    }

    public void setUseBIAS(boolean useBIAS) {
        this.useBIAS = useBIAS;
    }

    public NNLayers add(NNLayer.Builder builder) {
        NNLayer layer = builder.build(this, layers.size() == 0 ? 0 : layers.get(layers.size() - 1).neurons.length);
        layers.add(layer);

        return this;
    }

    public double[] feedForward(double[] input) {
        NNLayer inputs = layers.get(0);
        if (inputs.neurons.length == input.length) {
            inputs.setNeurons(input);
            for (int i = 1; i < layers.size(); i++) {
                layers.get(i).feed(layers.get(i - 1));
            }

            return layers.get(layers.size() - 1).neurons.clone();
        }

        return null;
    }

    public double cost(double[] expected) {
        return layers.get(layers.size() - 1).cost(expected);
    }

    public void backPropagate(double[] excepted) {
        for (int i = layers.size() - 1; i > 0; i--) {
            NNLayer prevLayer = layers.get(i - 1);
            NNLayer layer = layers.get(i);
            double[] neurons = layer.neurons;
            double[][] weights = layer.weights;
            double[][] localGradients = layer.localGradients;
            for (int j = 0; j < neurons.length; j++) {
                if (!layer.dropped[j]) {
                    double errorSignal = i == layers.size() - 1 ? excepted[j] - neurons[j] : 0;
                    double[] weight = weights[j];
                    for (int k = 0; k < weight.length; k++) {
                        if (!prevLayer.dropped[k]) {
                            if (i == layers.size() - 1) { //BACK PROP FOR OUTPUT LAYER
                                localGradients[j][k] = errorSignal * (layer.activation instanceof SoftMax ?
                                        ((SoftMax) layer.activation).derivative(rawResult(i, j), prevLayer.neurons, neurons)
                                        :
                                        layer.activation.derivative(rawResult(i, j)));
                            } else { //BACK PROP FOR HIDDEN LAYER
                                NNLayer nextLayer = layers.get(i + 1);
                                localGradients[j][k] = (layer.activation instanceof SoftMax ?
                                        ((SoftMax) layer.activation).derivative(rawResult(i, j), prevLayer.neurons, neurons)
                                        :
                                        layer.activation.derivative(rawResult(i, j))) * calculateNextLocalGradW(nextLayer);
                            }
                            double deltaWeight = learningRate * localGradients[j][k] * prevLayer.neurons[k];
                            weight[k] += deltaWeight;
                        }
                    }
                    if (useBIAS) {
                        if (i == layers.size() - 1) { //CORRECT BIAS FOR OUTPUT LAYER
                            layer.biasGradient = errorSignal * (layer.activation instanceof SoftMax ?
                                    ((SoftMax) layer.activation).derivative(rawResult(i, j), prevLayer.neurons, neurons)
                                    :
                                    layer.activation.derivative(rawResult(i, j)));
                        } else { //CORRECT BIAS FOR HIDDEN LAYER
                            NNLayer nextLayer = layers.get(i + 1);
                            layer.biasGradient = (layer.activation instanceof SoftMax ?
                                    ((SoftMax) layer.activation).derivative(rawResult(i, j), prevLayer.neurons, neurons)
                                    :
                                    layer.activation.derivative(rawResult(i, j))) * calculateNextLocalGradW(nextLayer);
                        }
                        layer.bias += learningRate * layer.biasGradient;
                    }
                }
            }
        }
    }

    private double calculateNextLocalGradW(NNLayer nextLayer) {
        double nextLocalGradW = 0;
        for (int l = 0; l < nextLayer.weights.length; l++) {
            for (int m = 0; m < nextLayer.weights[l].length; m++) {
                nextLocalGradW += nextLayer.weights[l][m] * nextLayer.localGradients[l][m];
            }
        }
        if (useBIAS) nextLocalGradW += nextLayer.bias * nextLayer.biasGradient;
        return nextLocalGradW;
    }

    protected double rawResult(int layerNum, int neuronNum) {
        NNLayer layer = layers.get(layerNum);
        NNLayer prevLayer = layers.get(layerNum - 1);
        double result = 0;
        for (int i = 0; i < prevLayer.neurons.length; i++) {
            if (!prevLayer.dropped[i]) result += prevLayer.neurons[i] * layer.weights[neuronNum][i];
        }
        if (useBIAS) result += prevLayer.bias;

        return result;
    }

}
