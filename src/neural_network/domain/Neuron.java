package neural_network.domain;

import java.io.Serializable;
import java.util.function.Function;

import neural_network.interfaces.SerializableFunction;

public class Neuron implements Serializable {
    private static final long serialVersionUID = 1L;

    private float value;
    private float bias;
    private float[] weight;
    private SerializableFunction<Float, Float> activationFunction;
    private SerializableFunction<Float, Float> activationDerivative;

    public Neuron(int numOfPreviousLayerNodes, SerializableFunction<Float, Float> activationFunction,
            SerializableFunction<Float, Float> activationDerivative) {
        this.activationFunction = activationFunction;
        this.activationDerivative = activationDerivative;

        this.bias = (float) (Math.random() * 0.1);
        this.weight = new float[numOfPreviousLayerNodes];

        if (numOfPreviousLayerNodes != 0) {
            float stddev = (float) (Math.sqrt(1.0 / numOfPreviousLayerNodes));
            for (int i = 0; i < weight.length; i++) {
                weight[i] = (float) (Math.random() * stddev);
            }
        }
    }

    public void activate(float input) {
        this.value = activationFunction.apply(input);
    }

    public float getValue() {
        return value;
    }

    public float getBias() {
        return bias;
    }

    public float getWeight(int index) {
        return weight[index];
    }

    public float[] getWeights() {
        return weight;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public void setWeight(float weight, int index) {
        this.weight[index] = weight;
    }

    public Function<Float, Float> getActivationFunction() {
        return activationFunction;
    }

    public Function<Float, Float> getActivationDerivative() {
        return activationDerivative;
    }

}
