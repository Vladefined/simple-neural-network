package ru.vladefined.neuralnetwork.modules;

import ru.vladefined.neuralnetwork.modules.NNDataSet;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

public class NNUtils {

    public static double[] RGBImageToVector(BufferedImage image) {
        double[] result = new double[image.getWidth() * image.getHeight() * 3];
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                int rgb = image.getRGB(x, y);
                result[x + y * image.getWidth()] = (new Color(rgb).getRed() / 255.0) * 2 - 1.0;
                result[x + y * image.getWidth() + 1] = (new Color(rgb).getGreen() / 255.0) * 2 - 1.0;
                result[x + y * image.getWidth() + 2] = (new Color(rgb).getBlue() / 255.0) * 2 - 1.0;
            }
        }

        return result;
    }

    public static NNDataSet parseMNISTFromCSV(String path, int max) throws IOException {
        String raw = Files.readString(new File(path).toPath());
        String[] lines = raw.split("\r\n");
        double[][] inputs = new double[Math.min(lines.length - 1, max)][];
        double[][] outputs = new double[Math.min(lines.length - 1, max)][];
        for (int i = 1; i < Math.min(lines.length, max + 1); i++) {
            String[] vals = lines[i].split(",");
            inputs[i - 1] = new double[vals.length - 1];
            outputs[i - 1] = numToVector(Integer.parseInt(vals[0]), 10);
            for (int j = 1; j < vals.length; j++) {
                inputs[i - 1][j - 1] = Integer.parseInt(vals[j]) / 255.0;
            }
        }

        return new NNDataSet(inputs, outputs);
    }

    public static double[] numToVector(int num, int length) {
        double[] vector = new double[length];
        vector[num] = 1.0;
        return vector;
    }

    public static int vectorToNum(double[] vector) {
        int num = 0;
        for (int i = 1; i < vector.length; i++) {
            if (vector[i] > vector[num]) num = i;
        }

        return num;
    }

}
