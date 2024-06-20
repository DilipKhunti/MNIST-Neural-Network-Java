import java.io.*;

public class NNetUtilities {

    private static final double ALPHA = 0.1;

    public MnistMatrix[] readMNISTData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(
                new BufferedInputStream(new FileInputStream(dataFilePath)));
        dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        DataInputStream labelInputStream = new DataInputStream(
                new BufferedInputStream(new FileInputStream(labelFilePath)));
        labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        MnistMatrix[] data = new MnistMatrix[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for (int i = 0; i < numberOfItems; i++) {
            MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols);
            mnistMatrix.setLabel(labelInputStream.readUnsignedByte());
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    mnistMatrix.setValue(r, c, dataInputStream.readUnsignedByte());
                }
            }
            data[i] = mnistMatrix;
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }

    public static void createLayerLog(final Neuron[] layer, String layerName, int numOfPreviousLayerNodes)
            throws IOException {
        String biasFileName = "log/" + layerName + "Bias.log";
        String weightFileame = "log/" + layerName + "Weight.log";

        PrintWriter biasPrintWriter = new PrintWriter(new FileWriter(biasFileName));
        PrintWriter weighPrintWriter = new PrintWriter(new FileWriter(weightFileame));

        for (Neuron value : layer) {
            biasPrintWriter.println(Double.toString(value.getBias()));

            for (int i = 0; i < numOfPreviousLayerNodes; i++) {
                weighPrintWriter.println(Double.toString(value.getWeight(i)));
            }
        }

        System.out.println(biasFileName + " created successfully!");
        System.out.println(weightFileame + " created successfully!");

        biasPrintWriter.close();
        weighPrintWriter.close();
    }

    public static void initLayerFromLog(Neuron[] layer, String layerName, int numOfPreviousLayerNodes)
            throws IOException {

        String biasPath = "log/" + layerName + "Bias.log";
        String weightPath = "log/" + layerName + "Weight.log";

        BufferedReader biasBufferedReader = new BufferedReader(new FileReader(biasPath));
        BufferedReader weightBufferedReader = new BufferedReader(new FileReader(weightPath));

        for (Neuron value : layer) {
            String biasLine = biasBufferedReader.readLine();

            value.setBias(Double.parseDouble(biasLine));

            for (int i = 0; i < numOfPreviousLayerNodes; i++) {
                String WeightLine = weightBufferedReader.readLine();
                value.setWeight(Double.parseDouble(WeightLine), i);
            }

        }

        System.out.println(biasPath + "data readed successfully!");
        System.out.println(weightPath + "data readed successfully!");
        biasBufferedReader.close();
        weightBufferedReader.close();

    }

    public static double sigmoid(double x) {
        return (1 / (1 + Math.exp(-x)));
    }

    public static double dSigmoid(double x) {
        return (x * (1 - x));
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double drelu(double x) {
        return x > 0 ? 1 : 0;
    }

    public static double leakyReLU(double x) {
        return (x > 0) ? x : ALPHA * x;
    }

    public static double dLeakyReLU(double x) {
        return (x > 0) ? 1.0 : ALPHA;
    }

    public static double elu(double x) {
        return x > 0 ? x : ALPHA * (Math.exp(x) - 1);
    }

    public static double delu(double x) {
        return x > 0 ? 1 : ALPHA * Math.exp(x);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double dtanh(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }

}
