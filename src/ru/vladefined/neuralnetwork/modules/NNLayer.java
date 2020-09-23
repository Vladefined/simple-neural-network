package ru.vladefined.neuralnetwork.modules;

import ru.vladefined.neuralnetwork.activation.*;
import ru.vladefined.neuralnetwork.weightinitialization.WeightInit;

public class NNLayer {
    public NNActivation activation;

    public double[] neurons;
    public double[][] weights, lastDeltaWeight;
    public double dropout = 0;

    public double bias, lastDeltaBias;

    private NNLayers layers;

    protected NNLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        this.neurons = new double[neurons];
        weights = new double[neurons][prevLayerNeurons];
        lastDeltaWeight = new double[neurons][prevLayerNeurons];
        this.layers = layers;
    }

    protected NNLayer activation(NNActivation activation) {
        this.activation = activation;

        return this;
    }

    protected NNLayer weightInit(WeightInit weightInitialization) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = weightInitialization.initWeight(layers.layers.get(layers.layers.size() - 1).neurons.length, neurons.length);
            }
        }
        bias = weightInitialization.initWeight(layers.layers.get(layers.layers.size() - 1).neurons.length, neurons.length);

        return this;
    }

    protected NNLayer dropout(double d) {
        dropout = d;

        return this;
    }

    public void setNeurons(double[] neurons) {
        this.neurons = neurons.clone();
    }

    public double calculateGradient(double neuron, double WGSum) {
        return activation.derivative(neuron) * WGSum;
    }

    public double weightsGradientsSum(double[][] localGradients, double biasGradient) {
        double nextLocalGradW = 0;
        for (int l = 0; l < weights.length; l++) {
            for (int m = 0; m < weights[l].length; m++) {
                nextLocalGradW += weights[l][m] * localGradients[l][m];
            }
        }
        if (layers.useBIAS) nextLocalGradW += bias * biasGradient;
        return nextLocalGradW;
    }

    public void feed(NNLayer previous, boolean isLearning) {
        double[] previousNeurons = previous.neurons;
        for (int i = 0; i < neurons.length; i++) {
            double sum = 0;
            for (int j = 0; j < previousNeurons.length; j++) {
                sum += weights[i][j] * previousNeurons[j];
            }
            neurons[i] = activation instanceof SoftMax ?
                    ((SoftMax) activation).activate(sum + (layers.useBIAS ? previous.bias : 0), neurons)
                    :
                    activation.activate(sum + (layers.useBIAS ? previous.bias : 0));
        }
    }

    protected static class Builder {
        protected int neurons;
        protected NNActivation activation = NNActivation.TANH;
        protected WeightInit weightInitialization = WeightInit.ONES;
        protected double d = 0.0;

        public Builder(int neurons) {
            this.neurons = neurons;
        }

        protected NNLayer build(NNLayers layers, int prevLayerNeurons) {
            NNLayer layer = new NNLayer(layers, prevLayerNeurons, neurons);
            layer.activation(activation);
            layer.weightInit(weightInitialization);
            layer.dropout(d);

            return layer;
        }

        protected Builder weightInit(WeightInit weightInitialization) {
            this.weightInitialization = weightInitialization;

            return this;
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
