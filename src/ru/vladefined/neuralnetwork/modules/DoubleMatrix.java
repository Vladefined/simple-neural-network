package ru.vladefined.neuralnetwork.modules;

import java.util.Arrays;
import java.util.Locale;

public class DoubleMatrix {
    private double[][] matrix;
    private int w, h;

    public DoubleMatrix(double[][] matrix) {
        Locale.setDefault(Locale.ENGLISH);
        if (matrix.length == 0) throw new RuntimeException("Matrix cannot be empty!");
        int maxLength = 0;
        for (double[] doubles : matrix) {
            if (doubles.length > maxLength) maxLength = doubles.length;
        }
        w = maxLength;
        h = matrix.length;
        double[][] finalMatrix = new double[matrix.length][maxLength];
        for (int i = 0; i < matrix.length; i++) {
            System.arraycopy(matrix[i], 0, finalMatrix[i], 0, matrix[i].length);
        }
        this.matrix = finalMatrix;
    }

    public DoubleMatrix(int w, int h) {
        this(new double[h][w]);
    }

    public int getW() {
        return w;
    }

    public int getH() {
        return h;
    }

    public double[][] getMatrix() {
        return matrix;
    }

    public static DoubleMatrix createTo2DMatrix(int width, double... vector) {
        double[][] matrix = new double[(int) Math.ceil(vector.length / (double) width)][width];
        for (int i = 0; i < vector.length; i++) {
            matrix[i / width][i % width] = vector[i];
        }

        return new DoubleMatrix(matrix);
    }

    public static DoubleMatrix create(double val) {
        return new DoubleMatrix(new double[][]{{val}});
    }

    public static DoubleMatrix create(double... vector) {
        return new DoubleMatrix(new double[][]{vector});
    }

    public static DoubleMatrix random(int w, int h) {
        double[][] matrix = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                matrix[y][x] = Math.random();
            }
        }

        return new DoubleMatrix(matrix);
    }

    public static DoubleMatrix random(double rangeStart, double rangeEnd, int w, int h) {
        double[][] matrix = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                matrix[y][x] = Math.random() * (rangeEnd - rangeStart) + rangeStart;
            }
        }

        return new DoubleMatrix(matrix);
    }

    public double[][] to2DVector() {
        double[][] vector3d = new double[h][w];
        for (int i = 0; i < matrix.length; i++) {
            vector3d[i] = matrix[i].clone();
        }

        return vector3d;
    }

    public double[] toVector() {
        double[] vector = new double[h * w];
        for (int y = 0; y < h; y++) {
            System.arraycopy(matrix[y], 0, vector, y * w, w);
        }

        return vector;
    }

    public DoubleMatrix get(int x1, int y1, int x2, int y2) {
        int w = x2 - x1;
        int h = y2 - y1;
        double[][] matrix = new double[h][w];
        for (int y = 0; y < h; y++) {
            System.arraycopy(this.matrix[y], x1, matrix[y], 0, w);
        }

        return new DoubleMatrix(matrix);
    }

    public void set(DoubleMatrix matrix, int x, int y) {
        for (int i = y; i < Math.min(matrix.h, h); i++) {
            System.arraycopy(matrix.matrix[i], 0, this.matrix[i], x, Math.min(matrix.w, w));
        }
    }

    public void setRow(int y, double[] vector) {
        if (vector.length != w) throw new RuntimeException("Vector length must be same as matrix width!");
        matrix[y] = vector.clone();
    }

    public double get(int x, int y) {
        return matrix[y][x];
    }

    public void set(double val, int x, int y) {
        matrix[y][x] = val;
    }

    public DoubleMatrix div(DoubleMatrix doubleMatrix) {
        if (doubleMatrix.w != w && doubleMatrix.w != 1 && doubleMatrix.h != h && doubleMatrix.h != 1)
            throw new RuntimeException("Second's matrix shape must be same as first or weight or height must equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] / doubleMatrix.matrix[doubleMatrix.h == 1 ? 0 : y][doubleMatrix.w == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix sub(DoubleMatrix doubleMatrix) {
        if (doubleMatrix.w != w && doubleMatrix.w != 1 && doubleMatrix.h != h && doubleMatrix.h != 1)
            throw new RuntimeException("Second's matrix shape must be same as first or weight or height must equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] - doubleMatrix.matrix[doubleMatrix.h == 1 ? 0 : y][doubleMatrix.w == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix add(DoubleMatrix doubleMatrix) {
        if (doubleMatrix.w != w && doubleMatrix.w != 1 && doubleMatrix.h != h && doubleMatrix.h != 1)
            throw new RuntimeException("Second's matrix shape must be same as first or weight or height must equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] + doubleMatrix.matrix[doubleMatrix.h == 1 ? 0 : y][doubleMatrix.w == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix mul(DoubleMatrix doubleMatrix) {
        if (doubleMatrix.w != w && doubleMatrix.w != 1 && doubleMatrix.h != h && doubleMatrix.h != 1)
            throw new RuntimeException("Second's matrix shape must be same as first or weight or height must equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] * doubleMatrix.matrix[doubleMatrix.h == 1 ? 0 : y][doubleMatrix.w == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix div(double[] vector) {
        if (vector.length != w && vector.length != 1)
            throw new RuntimeException("Vector's width must be same as matrix or equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] / vector[vector.length == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix sub(double[] vector) {
        if (vector.length != w && vector.length != 1)
            throw new RuntimeException("Vector's width must be same as matrix or equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] - vector[vector.length == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix add(double[] vector) {
        if (vector.length != w && vector.length != 1)
            throw new RuntimeException("Vector's width must be same as matrix or equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] + vector[vector.length == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix mul(double[] vector) {
        if (vector.length != w && vector.length != 1)
            throw new RuntimeException("Vector's width must be same as matrix or equal 1!");

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] * vector[vector.length == 1 ? 0 : x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix div(double val) {

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] / val;
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix sub(double val) {

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] - val;
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix add(double val) {

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] + val;
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix mul(double val) {

        double[][] result = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][x] = matrix[y][x] * val;
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix getRowAsMatrix(int y) {
        return DoubleMatrix.create(matrix[y]);
    }

    public DoubleMatrix getColumnAsMatrix(int x) {
        double[] column = new double[h];
        for (int y = 0; y < h; y++) {
            column[y] = matrix[y][x];
        }

        return DoubleMatrix.create(column);
    }

    public double[] getRow(int y) {
        return matrix[y];
    }

    public double[] getColumn(int x) {
        double[] column = new double[h];
        for (int y = 0; y < h; y++) {
            column[y] = matrix[y][x];
        }

        return column;
    }

    public DoubleMatrix sumRows() {
        double[][] result = new double[h][1];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result[y][0] += matrix[y][x];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix dot(double val) {
        return mul(val).sumRows();
    }

    public DoubleMatrix dot(double[] vector) {
        return mul(vector).sumRows();
    }

    public DoubleMatrix dot(DoubleMatrix doubleMatrix) {
        return mul(doubleMatrix).sumRows();
    }

    public String toStringBeautify() {
        StringBuilder result = new StringBuilder();
        for (int y = 0; y < h; y++) {
            result.append("[");
            for (int x = 0; x < w; x++) {
                result.append(String.format("%.4f", matrix[y][x]));
                if (x != w - 1) result.append(", ");
            }
            result.append("]");
            if (y != h - 1) result.append("\n");
        }
        return result.toString();
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        if (w > 0) result.append(Arrays.toString(matrix[0]));
        for (int y = 1; y < h; y++) {
            result.append("\n").append(Arrays.toString(matrix[y]));
        }
        return result.toString();
    }
}
