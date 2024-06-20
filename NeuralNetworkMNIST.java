import java.io.*;

public class NeuralNetworkMNIST {

    private final static int NUM_INPUT_NODES = 784;
    private final static int NUM_HIDDEN_NODES = 784;
    private final static int NUM_OUTPUT_NODES = 10;

    private final static int NUM_OF_EPOCHS = 60000;
    private final static int NUM_TEST = 10000;
    private final static int LEARNING_ORDER = 5;
    // private final static double LEARNING_RATE = 0.1f;
    private static double LEARNING_RATE = 0.2f;

    private static Neuron[] hiddenLayer = new Neuron[NUM_HIDDEN_NODES];
    private static Neuron[] outputLayer = new Neuron[NUM_OUTPUT_NODES];

    public static boolean calculateNeuralNetworkOutput(MnistMatrix[] dataSet, int index) {
        for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
            double activation = hiddenLayer[i].getBias();
            int count = 0;

            for (int j = 0; j < dataSet[index].getNumberOfRows(); j++) {
                for (int k = 0; k < dataSet[index].getNumberOfColumns(); k++) {
                    activation += hiddenLayer[i].getWeight(count)
                            * (dataSet[index].getValue(j, k) / 255.0);
                    count++;
                }

            }

            hiddenLayer[i].setValue(NNetUtilities.sigmoid(activation));
        }

        for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
            double activation = outputLayer[i].getBias();

            for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
                activation += hiddenLayer[j].getValue() * outputLayer[i].getWeight(j);
            }

            outputLayer[i].setValue(NNetUtilities.sigmoid(activation));
        }

        int max = 0;
        for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
            max = outputLayer[i].getValue() > outputLayer[max].getValue() ? i : max;
        }
        System.out.print(
                index + "\tExpected Output : " + dataSet[index].getLabel() + "\tActual Output : "
                        + max);

        for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
            System.out.print("  ");
            System.out.printf("%2.4f", outputLayer[i].getValue());
        }
        System.out.println();

        return max == dataSet[index].getLabel();
    }

    public static void trainNeuralNetwork(MnistMatrix[] trainingDataSet) throws IOException {

        Neuron.initLayer(hiddenLayer, NUM_INPUT_NODES);
        Neuron.initLayer(outputLayer, NUM_HIDDEN_NODES);

        for (int ord = 0; ord < LEARNING_ORDER; ord++) {

            LEARNING_RATE /= 2.0;

            for (int epochs = 0; epochs < NUM_OF_EPOCHS; epochs++) {

                calculateNeuralNetworkOutput(trainingDataSet, epochs);

                double[] deltaOutput = new double[NUM_OUTPUT_NODES];
                for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
                    double errorInOutput = 0.0f;
                    if (i == trainingDataSet[epochs].getLabel()) {
                        errorInOutput = (1 - outputLayer[i].getValue());
                    } else {
                        errorInOutput = (0 - outputLayer[i].getValue());
                    }
                    deltaOutput[i] = errorInOutput * NNetUtilities.dSigmoid(outputLayer[i].getValue());
                }

                double[] deltaHidden = new double[NUM_HIDDEN_NODES];
                for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
                    double errorInHidden = 0.0f;
                    for (int j = 0; j < NUM_OUTPUT_NODES; j++) {
                        errorInHidden += deltaOutput[j] * outputLayer[j].getWeight(i);
                    }
                    deltaHidden[i] = errorInHidden * NNetUtilities.dSigmoid(hiddenLayer[i].getValue());
                }

                for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
                    double tempValue = outputLayer[i].getBias() + (deltaOutput[i] * LEARNING_RATE);
                    outputLayer[i].setBias(tempValue);

                    for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
                        tempValue = outputLayer[i].getWeight(j)
                                + (hiddenLayer[j].getValue() * deltaOutput[i] * LEARNING_RATE);
                        outputLayer[i].setWeight(tempValue, j);
                    }
                }

                for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
                    double tempValue = hiddenLayer[i].getBias() + (deltaHidden[i] * LEARNING_RATE);
                    hiddenLayer[i].setBias(tempValue);

                    int count = 0;

                    for (int j = 0; j < trainingDataSet[epochs].getNumberOfRows(); j++) {
                        for (int k = 0; k < trainingDataSet[epochs].getNumberOfColumns(); k++) {
                            tempValue = hiddenLayer[i].getWeight(count)
                                    + (trainingDataSet[epochs].getValue(j, k) * deltaHidden[i] * LEARNING_RATE);
                            hiddenLayer[i].setWeight(tempValue, count);
                            count++;
                        }
                    }
                }
            }
        }

        NNetUtilities.createLayerLog(hiddenLayer, "hidden", NUM_INPUT_NODES);
        NNetUtilities.createLayerLog(outputLayer, "output", NUM_HIDDEN_NODES);
    }

    public static int testNeuralNetwork(MnistMatrix[] testingDataSet) throws IOException {
        Neuron.initLayer(hiddenLayer, NUM_INPUT_NODES);
        NNetUtilities.initLayerFromLog(hiddenLayer, "hidden", NUM_INPUT_NODES);

        Neuron.initLayer(outputLayer, NUM_HIDDEN_NODES);
        NNetUtilities.initLayerFromLog(outputLayer, "output", NUM_HIDDEN_NODES);

        int accurate = 0;

        for (int i = 0; i < NUM_TEST; i++) {
            if (calculateNeuralNetworkOutput(testingDataSet, i)) {
                accurate++;
            }
        }

        return accurate;

    }
}
