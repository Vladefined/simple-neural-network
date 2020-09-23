package ru.vladefined.examples;

import ru.vladefined.neuralnetwork.lossfunction.LossFunction;
import ru.vladefined.neuralnetwork.modules.DoubleMatrix;
import ru.vladefined.neuralnetwork.modules.lstm.LSTMNetwork;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.Set;

public class LSTMTextGeneration {
    private static char[] chars;
    private static final int inputLen = 6, generatedLength = 60;

    public static void main(String[] args) {
        String text = "Simple LSTM text generation example. Hello world!";
        extractAllCharacters(text);
        LSTMNetwork network = new LSTMNetwork.Builder(inputLen * chars.length, chars.length)
                .learningRate(0.08)
                .lossFunction(LossFunction.L2)
                .build();
        DoubleMatrix[] io = getInputsAndOutputs(text);
        for (int i = 0; i < 2500; i++) {
            network.fit(io[0], io[1]);
            System.out.println("EPOCH " + i);
            System.out.println();
            System.out.println("COST - " + network.cost());
            System.out.println("===========================");
    }
        gen(text, network);
    }

    private static void gen(String text, LSTMNetwork network) {
        String start = text.substring(0, inputLen);
        System.out.print(start);
        for (int i = 0; i < generatedLength; i++) {
            char next = vectorToChar(network.feedForward(charsToVector(start.toCharArray())));
            System.out.print(next + (i == generatedLength - 1 ? "\n" : ""));
            start = start.substring(1) + next;
        }
    }

    private static DoubleMatrix[] getInputsAndOutputs(String text) {
        DoubleMatrix inputs = new DoubleMatrix(inputLen * chars.length, text.length() - inputLen);
        DoubleMatrix outputs = new DoubleMatrix(chars.length, text.length() - inputLen);
        for (int i = 0; i < text.length() - inputLen; i++) {
            String start = text.substring(i, i + inputLen);
            char nextChar = text.charAt(i + inputLen);
            inputs.setRow(i, charsToVector(start.toCharArray()));
            outputs.setRow(i, charToVector(nextChar));
        }

        return new DoubleMatrix[] {inputs, outputs};
    }

    private static char[] vectorToChars(double[] vec) {
        char[] chars = new char[vec.length / LSTMTextGeneration.chars.length];
        for (int i = 0; i < chars.length; i++) {
            double[] temp = new double[LSTMTextGeneration.chars.length];
            System.arraycopy(vec, i * LSTMTextGeneration.chars.length, temp, 0, LSTMTextGeneration.chars.length);
            chars[i] = vectorToChar(temp);
        }

        return chars;
    }

    private static double[] charsToVector(char[] chars) {
        double[] vec = new double[LSTMTextGeneration.chars.length * chars.length];
        for (int i = 0; i < chars.length; i++) {
            vec[i * LSTMTextGeneration.chars.length + getCharID(chars[i])] = 1.0;
        }

        return vec;
    }

    private static void extractAllCharacters(String str) {
        char[] chars = str.toCharArray();
        Set<Character> temp = new HashSet<>();
        for (char c : chars)
            temp.add(c);

        LSTMTextGeneration.chars = new char[temp.size()];
        int i = 0;
        for (char c : temp) {
            LSTMTextGeneration.chars[i++] = c;
        }
    }

    private static int getCharID(char c) {
        int i = 0;
        for (char ch : chars) {
            if (ch == c) return i;
            i++;
        }

        return -1;
    }

    private static double[] charToVector(char c) {
        double[] vec = new double[chars.length];
        vec[getCharID(c)] = 1.0;
        return vec;
    }

    private static char vectorToChar(double[] vec) {
        int max = 0;
        for (int i = 0; i < vec.length; i++) {
            if (vec[max] < vec[i]) max = i;
        }

        return chars[max];
    }

    private static String read(String file) throws IOException {
        return Files.readString(new File(file).toPath());
    }

}
