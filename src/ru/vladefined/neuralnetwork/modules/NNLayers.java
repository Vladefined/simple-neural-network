package ru.vladefined.neuralnetwork.modules;

import ru.vladefined.neuralnetwork.NeuralNetwork;
import ru.vladefined.neuralnetwork.layers.*;

import java.util.ArrayList;
import java.util.List;

public class NNLayers {
    private NeuralNetwork.Builder builder;
    public List<NNLayer> layers = new ArrayList<>();
    public double learningRate, momentum;
    public boolean useBIAS = false;

    public NNLayers(NeuralNetwork.Builder builder, double learningRate, double momentum) {
        this.builder = builder;
        this.learningRate = learningRate;
        this.momentum = momentum;
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
        return ((OutputLayer) layers.get(layers.size() - 1)).cost(expected);
    }

}
