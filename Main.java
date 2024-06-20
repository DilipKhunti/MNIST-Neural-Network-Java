import java.util.Scanner;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        Scanner input = new Scanner(System.in);
        int choice = 0;

        while (true) {
            System.out.println("[1] Train Neural Network for MNIST");
            System.out.println("[2] Test Neural Network for MNIST");
            System.out.println("[0] Exit Programme");
            System.out.print("\nEnter Choice :");
            choice = input.nextInt();

            switch (choice) {
                case 1:
                    MnistMatrix[] trainingDataSet = new NNetUtilities().readMNISTData("data/train-images.idx3-ubyte",
                            "data/train-labels.idx1-ubyte");
                    NeuralNetworkMNIST.trainNeuralNetwork(trainingDataSet);

                    break;
                case 2:
                    MnistMatrix[] testingDataSet = new NNetUtilities().readMNISTData("data/t10k-images.idx3-ubyte",
                            "data/t10k-labels.idx1-ubyte");
                    int accurate = NeuralNetworkMNIST.testNeuralNetwork(testingDataSet);

                    System.out.println("Accurate : " + accurate);

                    break;
                case 0:
                    System.out.println("\nExiting the Programme!");
                    input.close();
                    return;

                default:
                    System.out.println("\nInvalid Input!\n");
                    break;
            }
        }

    }
}
