package neural_network.service;

import java.io.Serializable;

import neural_network.domain.Layer;
import neural_network.exception.NeuronMismatchException;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    private float learningRate;

    private Layer inputLayer;
    private Layer[] hiddenLayers;
    private Layer outputLayer;

    private float activation;
    private float errorInOutput;
    private float errorLayer;
    private float tempValue;

    private float[] deltaOutput;
    private float[][] deltaHiddens;
    

    public NeuralNetwork(float learningRate, Layer inputLayer, Layer[] hiddenLayers, Layer outputLayer) {
        this.learningRate = learningRate;

        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;

        this.deltaOutput = new float[outputLayer.layerSize()];
        this.deltaHiddens = new float[hiddenLayers.length][];

        for (int i = 0; i < hiddenLayers.length; i++) {
            deltaHiddens[i] = new float[hiddenLayers[i].layerSize()];
        }
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float[] calculateNeuralNetworkOutput(double[] dataSet) throws NeuronMismatchException {

        initInputLayer(dataSet);

        calculateLayer(hiddenLayers[0], inputLayer);
        for (int i = 1; i < hiddenLayers.length; i++) {
            calculateLayer(hiddenLayers[i], hiddenLayers[i - 1]);
        }

        calculateLayer(outputLayer, hiddenLayers[hiddenLayers.length - 1]);

        return outputLayer.getNeuronValuesArray();
    }

    public void trainNeuralNetwork(double[] trainingData, double[] expectedOutput) {

        calculateDeltaOutput(expectedOutput);

        calculateDeltaLayer(hiddenLayers[hiddenLayers.length
                - 1], hiddenLayers.length
                        - 1,
                outputLayer, deltaOutput);

        for (int i = hiddenLayers.length - 2; i >= 0; i--) {
            calculateDeltaLayer(hiddenLayers[i], i, hiddenLayers[i + 1], deltaHiddens[i + 1]);
        }

        backPropogateLayer(hiddenLayers[0], inputLayer, deltaHiddens[0]);
        for (int i = 1; i < hiddenLayers.length; i++) {
            backPropogateLayer(hiddenLayers[i], hiddenLayers[i - 1], deltaHiddens[i]);
        }
        backPropogateLayer(outputLayer, hiddenLayers[hiddenLayers.length - 1], deltaOutput);
    }

    private void initInputLayer(double[] dataSet) throws NeuronMismatchException {
        if (inputLayer.layerSize() != dataSet.length) {
            throw new NeuronMismatchException("Size of dataset and input layer does not matches!");
        }

        for (int i = 0; i < inputLayer.layerSize(); i++) {
            inputLayer.getNeuron(i).setValue((float) dataSet[i]);
        }

    }

    private void calculateLayer(Layer currentLayer, Layer previousLayer) {
        for (int i = 0; i < currentLayer.layerSize(); i++) {
            activation = currentLayer.getNeuron(i).getBias();

            for (int j = 0; j < previousLayer.layerSize(); j++) {
                activation += previousLayer.getNeuron(j).getValue() * currentLayer.getNeuron(i).getWeight(j);
            }

            currentLayer.getNeuron(i)
                    .setValue(currentLayer.getNeuron(i).getActivationFunction().apply(activation));
        }
    }

    private void calculateDeltaOutput(double[] expectedOutput) {
        for (int i = 0; i < outputLayer.layerSize(); i++) {
            errorInOutput = (float) expectedOutput[i] - outputLayer.getNeuron(i).getValue();

            deltaOutput[i] = errorInOutput * outputLayer.getNeuron(i).getActivationDerivative()
                    .apply(outputLayer.getNeuron(i).getValue());
        }
    }

    private void calculateDeltaLayer(Layer currentLayer, int index, Layer nextLayer, float[] deltaNextLayer) {

        for (int i = 0; i < currentLayer.layerSize(); i++) {
            errorLayer = 0.0f;

            for (int j = 0; j < nextLayer.layerSize(); j++) {
                errorLayer += deltaNextLayer[j] * nextLayer.getNeuron(j).getWeight(i);
            }

            deltaHiddens[index][i] = errorLayer * currentLayer.getNeuron(i).getActivationDerivative()
                    .apply(currentLayer.getNeuron(i).getValue());
        }
    }

    private void backPropogateLayer(Layer currentLayer, Layer previousLayer, float[] deltaLayer) {
        for (int i = 0; i < currentLayer.layerSize(); i++) {
            tempValue = currentLayer.getNeuron(i).getBias() + (deltaLayer[i] * learningRate);
            currentLayer.getNeuron(i).setBias(tempValue);

            for (int j = 0; j < previousLayer.layerSize(); j++) {
                tempValue = currentLayer.getNeuron(i).getWeight(j)
                        + (previousLayer.getNeuron(j).getValue() * deltaLayer[i] * learningRate);
                currentLayer.getNeuron(i).setWeight(tempValue, j);
            }
        }
    }
}
