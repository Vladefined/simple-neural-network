package ru.vladefined.neuralnetwork.modules;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.activation.SoftMax;

public class NNLayer {
    protected NNActivation activation;

    protected double[] neurons;
    protected boolean[] dropped;
    protected double[][] weights, localGradients;

    protected double bias, biasGradient;

    private NNLayers layers;

    protected NNLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        this.neurons = new double[neurons];
        dropped = new boolean[neurons];
        weights = new double[neurons][prevLayerNeurons];
        localGradients = new double[neurons][prevLayerNeurons];
        this.layers = layers;
    }

    protected NNLayer activation(NNActivation activation) {
        this.activation = activation;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = activation.weightInitialization();
            }
        }
        bias = activation.weightInitialization();

        return this;
    }

    protected NNLayer dropout(double d) {
        if (d >= 1.0 || d <= 0.0) return this;
        for (int i = 0; i < dropped.length; i++) {
            if (Math.random() < d) dropped[i] = true;
        }

        return this;
    }

    public void setNeurons(double[] neurons) {
        this.neurons = neurons.clone();
    }

    public void feed(NNLayer previous) {
        double[] previousNeurons = previous.neurons;
        for (int i = 0; i < neurons.length; i++) {
            if (!dropped[i]) {
                double sum = 0;
                for (int j = 0; j < previousNeurons.length; j++) {
                    if (!previous.dropped[j]) sum += weights[i][j] * previousNeurons[j];
                }
                neurons[i] = activation instanceof SoftMax ?
                        ((SoftMax) activation).activate(sum + (layers.useBIAS ? previous.bias : 0), neurons)
                        :
                        activation.activate(sum + (layers.useBIAS ? previous.bias : 0));
            }
        }
    }

    public double cost(double[] expected) {
        double result = 0;
        for (int i = 0; i < neurons.length; i++) {
            result += (expected[i] - neurons[i]) * (expected[i] - neurons[i]);
        }

        return 1.0 / 2 * result;
    }

    protected static class Builder {
        protected int neurons;
        protected NNActivation activation = null;
        protected double d = 0.0;

        public Builder(int neurons) {
            this.neurons = neurons;
        }

        protected NNLayer build(NNLayers layers, int prevLayerNeurons) {
            NNLayer layer = new NNLayer(layers, prevLayerNeurons, neurons);
            if (activation != null) layer.activation(activation);
            if (d > 0.0 && d < 1.0) layer.dropout(d);

            return layer;
        }

        protected Builder activation(NNActivation activation) {
            this.activation = activation;

            return this;
        }

        protected Builder dropout(double d) {
            this.d = d;

            return this;
        }

    }

}
