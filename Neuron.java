public class Neuron {
    private double value;
    private double bias;
    private double[] weight;

    public Neuron(int numOfPreviousLayerNodes) {
        bias = Math.random() * 0.1;
        weight = new double[numOfPreviousLayerNodes];

        double stddev = Math.sqrt(1.0 / numOfPreviousLayerNodes);

        for (int i = 0; i < weight.length; i++) {
            weight[i] = Math.random() * stddev;
        }
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return this.value;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getBias() {
        return this.bias;
    }

    public void setWeight(double weight, int index) {
        this.weight[index] = weight;
    }

    public double getWeight(int index) {
        return this.weight[index];
    }

    public static void initLayer(Neuron[] layer, int numOfPreviousLayerNodes) {
        for (int i = 0; i < layer.length; i++) {
            layer[i] = new Neuron(numOfPreviousLayerNodes);
        }
    }

}
