package ru.vladefined.neuralnetwork.optimization;

import ru.vladefined.neuralnetwork.modules.NNDataSet;
import ru.vladefined.neuralnetwork.modules.NNIterationListener;
import ru.vladefined.neuralnetwork.modules.NNLayers;

public interface OptimizationAlgorithm {
    OptimizationAlgorithm
            STOCHASTIC_GRADIENT_DESCENT = new StochasticGradientDescent();

    void fit(NNLayers layers, NNDataSet dataSet);

    void setListener(NNIterationListener listener);

    double cost();

}
