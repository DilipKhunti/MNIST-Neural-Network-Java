import MNIST.domain.*;
import MNIST.reader.*;
import neural_network.reader.DataReader;
import neural_network.reader.SavedNetworkIO;
import neural_network.service.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.InputMismatchException;
import java.util.Scanner;

public class NeuralNetworkApp {
    private static final String TRAIN_IMAGES_PATH = "../data/train-images.idx3-ubyte";
    private static final String TRAIN_LABELS_PATH = "../data/train-labels.idx1-ubyte";
    private static final String TEST_IMAGES_PATH = "../data/t10k-images.idx3-ubyte";
    private static final String TEST_LABELS_PATH = "../data/t10k-labels.idx1-ubyte";

    private static final String NETWORK_DETAILS = "0.05 784-sigmoid 512-relu 256-relu 10-sigmoid";
    private static final float INITIAL_LEARNING_RATE = 0.1f;
    private static final int EPOCHS = 5;

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        NeuralNetwork mnist = null;

        try {
            mnist = initializeNeuralNetwork();
        } catch (Exception e) {
            System.err.println("Error initializing the neural network: " + e.getMessage());
            input.close();
            return;
        }

        while (true) {
            int choice = displayMenuAndGetChoice(input);

            try {
                switch (choice) {
                    case 1:
                        trainNetwork(mnist, INITIAL_LEARNING_RATE);
                        break;
                    case 2:
                        testNetwork(mnist);
                        break;
                    case 3:
                        testImage(input, mnist);
                        break;
                    case 0:
                        input.close();
                        exitProgram(input);
                        return;
                    default:
                        System.out.println("Invalid Input! Please try again.");
                }
            } catch (Exception e) {
                System.err.println("An error occurred: " + e.getMessage());
            }
        }
    }

    private static NeuralNetwork initializeNeuralNetwork() throws Exception {
        try {
            return DataReader.createNeuralNetwork(NETWORK_DETAILS);
        } catch (IOException e) {
            throw new Exception("Failed to create neural network: " + e.getMessage());
        }
    }

    private static int displayMenuAndGetChoice(Scanner input) {
        while (true) {
            try {
                System.out.println();
                System.out.println("[1] Train Neural Network for MNIST");
                System.out.println("[2] Test Neural Network for MNIST");
                System.out.println("[3] Test Neural Network for Image");
                System.out.println("[0] Exit Program");
                System.out.print("Enter Choice: ");
                return input.nextInt();
            } catch (InputMismatchException e) {
                System.out.println("Invalid input! Please enter a number.");
                input.next();
            }
        }
    }

    private static void trainNetwork(NeuralNetwork mnist, float learningRate) {
        try {
            MnistMatrix[] trainingDataSet = new MNISTDataReader().readMNISTData(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                mnist.setLearningRate(learningRate);

                for (MnistMatrix data : trainingDataSet) {
                    float[] output = mnist.calculateNeuralNetworkOutput(data.toArray());
                    System.out.print(epoch + 1 + "\t");
                    testAndPrintOutput(output, data.getLabel());

                    double[] expectedOutput = new double[output.length];
                    for (int j = 0; j < expectedOutput.length; j++) {
                        expectedOutput[j] = (j == data.getLabel()) ? 1 : 0;
                    }

                    mnist.trainNeuralNetwork(data.toArray(), expectedOutput);
                }

                learningRate /= 2;
            }

            SavedNetworkIO.createNeuralNetworkSaves(mnist, "mnist");

        } catch (FileNotFoundException e) {
            System.err.println("Training data file not found: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Error reading training data: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An error occurred during training: " + e.getMessage());
        }
    }

    private static void testNetwork(NeuralNetwork mnist) {
        try {
            MnistMatrix[] testingDataSet = new MNISTDataReader().readMNISTData(TEST_IMAGES_PATH, TEST_LABELS_PATH);
            int accurate = 0;

            mnist = SavedNetworkIO.initNeuralNetworkFromSaves(mnist, "mnist");

            for (MnistMatrix data : testingDataSet) {
                float[] output = mnist.calculateNeuralNetworkOutput(data.toArray());
                if (testAndPrintOutput(output, data.getLabel())) {
                    accurate++;
                }
            }
            System.out.println("Accuracy: " + accurate + " / " + testingDataSet.length);

        } catch (FileNotFoundException e) {
            System.err.println("Test data file not found: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Error reading test data: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An error occurred during testing: " + e.getMessage());
        }
    }

    private static void testImage(Scanner input, NeuralNetwork mnist) {
        try {
            mnist = SavedNetworkIO.initNeuralNetworkFromSaves(mnist, "mnist");

            System.out.print("Enter Digit Value: ");
            int value = input.nextInt();

            System.out.print("Enter Image Name: ");
            String img = input.next();
            String path = "../Images/" + img;

            MnistMatrix imageData = ImageMatrix.getImageMatrix(path, value);
            ImageMatrix.displayImage(imageData);

            float[] output = mnist.calculateNeuralNetworkOutput(imageData.toArray());
            testAndPrintOutput(output, value);

        } catch (InputMismatchException e) {
            System.err.println("Invalid input! Please enter a valid number.");
            input.next();
        } catch (Exception e) {
            System.err.println("An error occurred while testing the image: " + e.getMessage());
        }
    }

    private static boolean testAndPrintOutput(float[] output, int label) {
        int calculatedValue = 0;

        for (int j = 0; j < output.length; j++) {
            if (output[j] > output[calculatedValue]) {
                calculatedValue = j;
            }
        }

        System.out.print("Expected: " + label + "\tActual: " + calculatedValue);
        for (float v : output) {
            System.out.print("  ");
            System.out.printf("%2.4f", v);
        }
        System.out.println();

        return calculatedValue == label;
    }

    private static void exitProgram(Scanner input) {
        System.out.println("Exiting the Program!");
        input.close();
    }
}
