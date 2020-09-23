package ru.vladefined.neuralnetwork.modules.lstm;

import ru.vladefined.neuralnetwork.lossfunction.LossFunction;
import ru.vladefined.neuralnetwork.modules.DoubleMatrix;

public class LSTMNetwork {
    private LSTMLayer layer;

    protected LSTMNetwork(int cells, int inputs, double learningRate, LossFunction lossFunction) {
        layer = new LSTMLayer(cells, inputs, learningRate, lossFunction);
    }

    public void fit(DoubleMatrix inputs, DoubleMatrix expected) {
        layer.fit(inputs, expected);
        layer.reset();
    }

    public double cost() {
        return layer.cost;
    }

    public void reset() {
        layer.reset();
    }

    public double[] feedForward(double[] inputs) {
        return layer.feedForward(inputs);
    }

    public static class Builder {
        private int cells, inputs;
        private double learningRate = 0.01;
        private LossFunction lossFunction = LossFunction.L2;

        public Builder(int inputs, int outputs) {
            this.cells = outputs;
            this.inputs = inputs;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;

            return this;
        }

        public Builder lossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;

            return this;
        }

        public LSTMNetwork build() {
            return new LSTMNetwork(inputs, cells, learningRate, lossFunction);
        }
    }

}
