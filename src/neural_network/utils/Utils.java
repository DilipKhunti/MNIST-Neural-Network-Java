package neural_network.utils;

public class Utils {

    private static final float ALPHA = 0.1f;

    private static final int TABLE_SIZE = 1024;

    // Lookup tables
    private static final float[] sigmoidLookupTable = new float[TABLE_SIZE];
    private static final float[] tanhLookupTable = new float[TABLE_SIZE];
    private static final float[] reluLookupTable = new float[TABLE_SIZE];
    private static final float[] leakyReLULookupTable = new float[TABLE_SIZE];

    static {
        for (int i = 0; i < TABLE_SIZE; i++) {
            float x = (i / (float) (TABLE_SIZE - 1)) * 12f - 6f;

            // Sigmoid
            sigmoidLookupTable[i] = (float) (1.0 / (1.0 + Math.exp(-x)));

            // Tanh
            tanhLookupTable[i] = (float) Math.tanh(x);

            // ReLU (using table index for positive x, since ReLU is 0 for negative x)
            reluLookupTable[i] = Math.max(0, x);

            // Leaky ReLU (using table index for positive x, and alpha * x for negative x)
            leakyReLULookupTable[i] = x > 0 ? x : ALPHA * x;
        }
    }

    public static float sigmoid(float x) {
        int index = (int) ((x + 6f) * (TABLE_SIZE - 1) / 12f);
        index = Math.max(0, Math.min(TABLE_SIZE - 1, index));
        return sigmoidLookupTable[index];
    }

    public static float dSigmoid(float x) {
        return (x * (1 - x));
    }

    public static float tanh(float x) {
        int index = (int) ((x + 6f) * (TABLE_SIZE - 1) / 12f);
        index = Math.max(0, Math.min(TABLE_SIZE - 1, index));
        return tanhLookupTable[index];
    }

    public static float dtanh(float x) {
        return 1 - x * x;
    }

    public static float relu(float x) {
        if (x < -6f)
            return 0;
        if (x > 6f)
            return x;
        int index = (int) ((x + 6f) * (TABLE_SIZE - 1) / 12f);
        return reluLookupTable[index];
    }

    public static float drelu(float x) {
        return x > 0 ? 1 : 0;
    }

    public static float leakyReLU(float x) {
        if (x < -6f)
            return ALPHA * x;
        if (x > 6f)
            return x;
        int index = (int) ((x + 6f) * (TABLE_SIZE - 1) / 12f);
        return leakyReLULookupTable[index];
    }

    public static float dLeakyReLU(float x) {
        return (x > 0) ? 1.0f : ALPHA;
    }

    public static float elu(float x) {
        return x > 0 ? x : ALPHA * (float) (Math.exp(x) - 1);
    }

    public static float delu(float x) {
        return x > 0 ? 1 : ALPHA * (float) Math.exp(x);
    }
}
