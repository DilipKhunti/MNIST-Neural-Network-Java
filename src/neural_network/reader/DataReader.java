package neural_network.reader;

import neural_network.domain.Layer;
import neural_network.exception.InvalidActivationFunctionException;
import neural_network.interfaces.SerializableFunction;
import neural_network.service.NeuralNetwork;
import neural_network.utils.Utils;

import static neural_network.utils.Constants.SPACE;
import static neural_network.utils.Constants.DASH;

import java.util.Arrays;

public class DataReader {
    private static final SerializableFunction<Float, Float> sigmoid = Utils::sigmoid;
    private static final SerializableFunction<Float, Float> dSigmoid = Utils::dSigmoid;

    private static final SerializableFunction<Float, Float> relu = Utils::relu;
    private static final SerializableFunction<Float, Float> drelu = Utils::drelu;

    private static final SerializableFunction<Float, Float> leakyReLU = Utils::leakyReLU;
    private static final SerializableFunction<Float, Float> dLeakyReLU = Utils::dLeakyReLU;

    private static final SerializableFunction<Float, Float> elu = Utils::elu;
    private static final SerializableFunction<Float, Float> delu = Utils::delu;

    private static final SerializableFunction<Float, Float> tanh = Utils::tanh;
    private static final SerializableFunction<Float, Float> dtanh = Utils::dtanh;

    public static NeuralNetwork createNeuralNetwork(String input) throws Exception {
        String[] neuralNetworkDetails = input.split(SPACE);

        float learningRate = Float.parseFloat(neuralNetworkDetails[0]);
        int inputNodes = Integer.parseInt(neuralNetworkDetails[1].split(DASH)[0]);
        int outputNodes = Integer.parseInt(neuralNetworkDetails[neuralNetworkDetails.length - 1].split(DASH)[0]);

        String[] hiddenLayersDeatails = Arrays.copyOfRange(neuralNetworkDetails, 2, neuralNetworkDetails.length - 1);

        Layer inputLayer = new Layer(inputNodes, 0, getActivationFunction(neuralNetworkDetails[1].split(DASH)[1]),
                getActivationDerivative(
                        neuralNetworkDetails[1].split(DASH)[1]));

        Layer[] hiddenLayers = new Layer[hiddenLayersDeatails.length];
        hiddenLayers[0] = new Layer(Integer.parseInt(hiddenLayersDeatails[0].split(DASH)[0]), inputLayer.layerSize(),
                getActivationFunction(hiddenLayersDeatails[0].split(DASH)[1]),
                getActivationDerivative(hiddenLayersDeatails[0].split(DASH)[1]));
                
        for (int i = 1; i < hiddenLayersDeatails.length; i++) {
            hiddenLayers[i] = new Layer(Integer.parseInt(hiddenLayersDeatails[i].split(DASH)[0]),
                    hiddenLayers[i - 1].layerSize(), getActivationFunction(hiddenLayersDeatails[i].split(DASH)[1]),
                    getActivationDerivative(hiddenLayersDeatails[i].split(DASH)[1]));
        }

        Layer outputLayer = new Layer(outputNodes, hiddenLayers[hiddenLayers.length - 1].layerSize(),
                getActivationFunction(neuralNetworkDetails[neuralNetworkDetails.length
                        - 1].split(DASH)[1]),
                getActivationDerivative(
                        neuralNetworkDetails[neuralNetworkDetails.length - 1].split(DASH)[1]));

        return new NeuralNetwork(learningRate, inputLayer, hiddenLayers, outputLayer);
    }

    private static SerializableFunction<Float, Float> getActivationFunction(String functionName)
            throws InvalidActivationFunctionException {
        if (functionName.equals("sigmoid")) {
            return sigmoid;
        } else if (functionName.equals("relu")) {
            return relu;
        } else if (functionName.equals("leakyrelu")) {
            return leakyReLU;
        } else if (functionName.equals("elu")) {
            return elu;
        } else if (functionName.equals("tanh")) {
            return tanh;
        }
        throw new InvalidActivationFunctionException("Invalid Activation Function!!");
    }

    private static SerializableFunction<Float, Float> getActivationDerivative(String functionName)
            throws InvalidActivationFunctionException {
        if (functionName.equals("sigmoid")) {
            return dSigmoid;
        } else if (functionName.equals("relu")) {
            return drelu;
        } else if (functionName.equals("leakyrelu")) {
            return dLeakyReLU;
        } else if (functionName.equals("elu")) {
            return delu;
        } else if (functionName.equals("tanh")) {
            return dtanh;
        }
        throw new InvalidActivationFunctionException("Invalid Activation Function!!");
    }
}
