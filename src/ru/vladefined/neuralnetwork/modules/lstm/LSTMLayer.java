package ru.vladefined.neuralnetwork.modules.lstm;

import ru.vladefined.neuralnetwork.lossfunction.LossFunction;
import ru.vladefined.neuralnetwork.modules.DoubleMatrix;

public class LSTMLayer {
    //TODO: Make as NNLayer

    private LSTMCell[] cells;
    protected LossFunction lossFunction;
    protected double learningRate;
    protected double cost = 0;

    public LSTMLayer(int inputs, int cells, double learningRate, LossFunction lossFunction) {
        this.learningRate = learningRate;
        this.cells = new LSTMCell[cells];
        for (int i = 0; i < cells; i++) {
            this.cells[i] = new LSTMCell(inputs, this);
        }
        this.lossFunction = lossFunction;
    }

    protected void reset() {
        for (LSTMCell cell : cells) {
            cell.prepare(1);
        }
    }

    protected double[] feedForward(double[] inputs) {
        double[] output = new double[cells.length];
        for (int i = 0; i < cells.length; i++) {
            output[i] = cells[i].forward(inputs);
        }

        return output;
    }

    protected void fit(DoubleMatrix inputs, DoubleMatrix expected) {
        double cost = 0;
        for (int i = 0; i < cells.length; i++) {
            cells[i].forwardLearning(inputs);
            cost += cells[i].backwardLearning(expected.getColumn(i));
        }
        this.cost = cost / cells.length;
    }

}
