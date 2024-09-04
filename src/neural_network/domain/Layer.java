package neural_network.domain;

import java.io.Serializable;

import neural_network.interfaces.SerializableFunction;

public class Layer implements Serializable {
    private static final long serialVersionUID = 1L;

    private Neuron[] neurons;
    private int numberOfNeurons;

    public Layer(int numberOfNeurons, int numOfPreviousLayerNodes,
            SerializableFunction<Float, Float> activationFunction,
            SerializableFunction<Float, Float> activationDerivative) {
        this.numberOfNeurons = numberOfNeurons;
        this.neurons = new Neuron[numberOfNeurons];
        initLayer(neurons, numOfPreviousLayerNodes, activationFunction, activationDerivative);
    }

    public Neuron getNeuron(int index) {
        return this.neurons[index];
    }

    public int layerSize() {
        return this.numberOfNeurons;
    }

    public void initLayer(Neuron[] layer, int numOfPreviousLayerNodes,
            SerializableFunction<Float, Float> activationFunction,
            SerializableFunction<Float, Float> activationDerivative) {
        for (int i = 0; i < layer.length; i++) {
            layer[i] = new Neuron(numOfPreviousLayerNodes, activationFunction, activationDerivative);
        }
    }

    public float[] getNeuronValuesArray() {
        float[] values = new float[numberOfNeurons];

        for (int i = 0; i < numberOfNeurons; i++) {
            values[i] = neurons[i].getValue();
        }

        return values;
    }
}
