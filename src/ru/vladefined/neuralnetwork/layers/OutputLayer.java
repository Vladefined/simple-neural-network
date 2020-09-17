package ru.vladefined.neuralnetwork.layers;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.modules.NNLayer;
import ru.vladefined.neuralnetwork.modules.NNLayers;

public class OutputLayer extends NNLayer {

    protected OutputLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        super(layers, prevLayerNeurons, neurons);
    }

    public static class Builder extends NNLayer.Builder {

        public Builder(int neurons) {
            super(neurons);
        }

        @Override
        public OutputLayer.Builder activation(NNActivation activation) {
            super.activation(activation);

            return this;
        }

        @Override
        protected OutputLayer build(NNLayers layers, int prevLayerNeurons) {
            OutputLayer layer = new OutputLayer(layers, prevLayerNeurons, this.neurons);
            if (activation != null) layer.activation(activation);
            if (d > 0.0 && d < 1.0) layer.dropout(d);

            return layer;
        }
    }

}
