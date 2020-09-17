package ru.vladefined.neuralnetwork.modules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NNDataSet {
    protected List<double[]> inputs = new ArrayList<>();
    protected List<double[]> outputs = new ArrayList<>();

    public NNDataSet() {

    }

    public NNDataSet(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length) return;
        this.inputs.addAll(Arrays.asList(inputs));
        this.outputs.addAll(Arrays.asList(outputs));
    }

    public void add(double[] input, double[] output) {
        inputs.add(input);
        outputs.add(output);
    }

    public int getInputLength() {
        return inputs.get(0).length;
    }

    public int getOutputLength() {
        return outputs.get(0).length;
    }

    public int size() {
        return inputs.size();
    }

    public double[][] get(int i) {
        return new double[][] {inputs.get(i), outputs.get(i)};
    }

    public void remove(int i) {
        inputs.remove(i);
        outputs.remove(i);
    }

    public void clear() {
        inputs.clear();
        outputs.clear();
    }

    public void set(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length) return;
        this.inputs.addAll(Arrays.asList(inputs));
        this.outputs.addAll(Arrays.asList(outputs));
    }

    public void shuffle() {
        for (int i = 0; i < inputs.size(); i++) {
            double[] ie = inputs.get(i);
            double[] oe = outputs.get(i);
            int ri = (int) Math.floor(Math.random() * inputs.size());
            inputs.set(i, inputs.get(ri));
            inputs.set(ri, ie);
            outputs.set(i, outputs.get(ri));
            outputs.set(ri, oe);
        }
    }

}
