package ru.vladefined.neuralnetwork.modules;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.activation.SoftMax;

public class NNLayer {
    protected NNActivation activation;

    protected double[] neurons;
    protected double[][] weights, localGradients;

    protected double bias, biasGradient;

    private NNLayers layers;

    protected NNLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        this.neurons = new double[neurons];
        weights = new double[neurons][prevLayerNeurons];
        localGradients = new double[neurons][prevLayerNeurons];
        this.layers = layers;
    }

    public NNLayer activation(NNActivation activation) {
        this.activation = activation;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = activation.weightInitialization();
            }
        }
        bias = activation.weightInitialization();

        return this;
    }

    public NNLayers next() {
        return layers;
    }

    public void setNeurons(double[] neurons) {
        this.neurons = neurons.clone();
    }

    public void feed(NNLayer previous) {
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

    public double cost(double[] expected) {
        double result = 0;
        for (int i = 0; i < neurons.length; i++) {
            result += (expected[i] - neurons[i]) * (expected[i] - neurons[i]);
        }

        return 1.0 / 2 * result;
    }

}
