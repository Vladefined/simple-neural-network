package ru.vladefined.neuralnetwork.layers;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.modules.NNLayer;
import ru.vladefined.neuralnetwork.modules.NNLayers;

public class HiddenLayer extends NNLayer {

    protected HiddenLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        super(layers, prevLayerNeurons, neurons);
    }

    public static class Builder extends NNLayer.Builder {
        public Builder(int neurons) {
            super(neurons);
        }

        @Override
        public HiddenLayer.Builder activation(NNActivation activation) {
            super.activation(activation);

            return this;
        }

        public HiddenLayer.Builder dropout(double d) {
            super.dropout(d);

            return this;
        }

        @Override
        protected HiddenLayer build(NNLayers layers, int prevLayerNeurons) {
            HiddenLayer layer = new HiddenLayer(layers, prevLayerNeurons, this.neurons);
            if (activation != null) layer.activation(activation);
            if (d > 0.0 && d < 1.0) layer.dropout(d);

            return layer;
        }
    }

}
