package ru.vladefined.neuralnetwork.layers;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.activation.SoftMax;
import ru.vladefined.neuralnetwork.lossfunction.LossFunction;
import ru.vladefined.neuralnetwork.modules.NNLayer;
import ru.vladefined.neuralnetwork.modules.NNLayers;
import ru.vladefined.neuralnetwork.weightinitialization.WeightInit;

public class OutputLayer extends NNLayer {
    public LossFunction lossFunction;

    protected OutputLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        super(layers, prevLayerNeurons, neurons);
        activation = NNActivation.SIGMOID;
    }

    public double calculateGradient(double errorSignal, double neuron, NNLayer prevLayer) {
        return errorSignal * (activation instanceof SoftMax ?
                ((SoftMax) activation).derivative(neuron, prevLayer.neurons)
                :
                activation.derivative(neuron));
    }

    public double cost(double[] expected) {
        return lossFunction.calculate(neurons, expected);
    }

    public static class Builder extends NNLayer.Builder {
        protected LossFunction lossFunction = LossFunction.L2;

        public Builder(int neurons) {
            super(neurons);
        }

        @Override
        public OutputLayer.Builder activation(NNActivation activation) {
            super.activation(activation);

            return this;
        }

        public OutputLayer.Builder lossFunc(LossFunction lossFunction) {
            this.lossFunction = lossFunction;

            return this;
        }

        public OutputLayer.Builder weightInit(WeightInit weightInitialization) {
            super.weightInit(weightInitialization);

            return this;
        }

        @Override
        public NNLayer.Builder dropout(double d) {
            return super.dropout(d);
        }

        @Override
        protected OutputLayer build(NNLayers layers, int prevLayerNeurons) {
            OutputLayer layer = new OutputLayer(layers, prevLayerNeurons, this.neurons);
            layer.activation(activation);
            layer.weightInit(weightInitialization);
            layer.dropout(d);
            layer.lossFunction = lossFunction;

            return layer;
        }
    }

}
