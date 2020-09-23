package ru.vladefined.neuralnetwork.layers;

import ru.vladefined.neuralnetwork.activation.NNActivation;
import ru.vladefined.neuralnetwork.modules.NNLayer;
import ru.vladefined.neuralnetwork.modules.NNLayers;

public class InputLayer extends NNLayer {

    protected InputLayer(NNLayers layers, int prevLayerNeurons, int neurons) {
        super(layers, prevLayerNeurons, neurons);
    }

    public static class Builder extends NNLayer.Builder {

        public Builder(int neurons) {
            super(neurons);
        }

        @Override
        protected InputLayer build(NNLayers layers, int prevLayerNeurons) {
            InputLayer layer = new InputLayer(layers, prevLayerNeurons, this.neurons);
            layer.activation(activation);
            layer.dropout(d);

            return layer;
        }
    }

}
