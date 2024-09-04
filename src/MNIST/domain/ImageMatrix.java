package MNIST.domain;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageMatrix {

    public static MnistMatrix getImageMatrix(String path, int value) {
        MnistMatrix matrix = new MnistMatrix(28, 28);

        BufferedImage originalImage = readImage(path);
        BufferedImage resizedImage = resizeImage(originalImage);

        matrix.setLabel(value);

        int newWidth = 28;
        int newHeight = 28;

        int[] pixels = resizedImage.getRGB(0, 0, newWidth, newHeight, null, 0, newWidth);

        for (int i = 0; i < newHeight; i++) {
            for (int j = 0; j < newWidth; j++) {

                int p = pixels[i * newWidth + j];

                int a = (p >> 24) & 0xff;
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;

                int avg = (r + g + b) / 3;

                p = (a << 24) | (avg << 16) | (avg << 8) | avg;

                pixels[i * newWidth + j] = p;
                if ((255 - avg) > 20) {
                    matrix.setValue(i, j, 255 - avg);

                } else {
                    matrix.setValue(i, j, 0);

                }

            }
        }

        resizedImage.setRGB(0, 0, newWidth, newHeight, pixels, 0, newWidth);

        return matrix;
    }

    private static BufferedImage readImage(String path) {
        BufferedImage originalImage = null;
        File inputFile = null;

        try {
            inputFile = new File(path);
            originalImage = ImageIO.read(inputFile);
        } catch (IOException e) {
            System.out.println("here");
            System.out.println(e);
        }

        return originalImage;
    }

    private static BufferedImage resizeImage(BufferedImage originalImage) {
        BufferedImage resizedImage = null;

        int newWidth = 28;
        int newHeight = 28;
        try {

            resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB);

            Graphics2D g = resizedImage.createGraphics();
            g.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
            g.dispose();

        } catch (Exception e) {
            System.out.println("here 1");
            System.out.println("Error: " + e.getMessage());
        }
        return resizedImage;
    }

    public static void displayImage(MnistMatrix imageData) {
        System.out.println("label: " + imageData.getLabel());
        for (int r = 0; r < imageData.getNumberOfRows(); r++) {
            for (int c = 0; c < imageData.getNumberOfColumns(); c++) {
                if (imageData.getValue(r, c) != 0) {

                    System.out.printf("%3d", imageData.getValue(r, c));
                } else
                    System.out.print("   ");
            }
            System.out.println();
        }
    }
}
