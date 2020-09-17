package ru.vladefined.neuralnetwork;

import java.awt.*;
import java.awt.image.BufferedImage;

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

}
