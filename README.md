# Neural Network for MNIST Handwritten Digit Recognition

This project implements a simple neural network for recognizing handwritten digits from the MNIST dataset. The network is trained using backpropagation and supports various activation functions like sigmoid, ReLU, leaky ReLU, ELU, and tanh. The network can be trained, tested on the MNIST dataset, and used to recognize custom images of handwritten digits.

## Features

- **Modular Design**: The code is organized into packages, making it easy to understand and extend.
- **Multiple Activation Functions**: The network supports sigmoid, ReLU, leaky ReLU, ELU, and tanh activation functions.
- **Image Processing**: Custom images can be processed and tested against the trained model.
- **Persistence**: The trained network can be saved and loaded for later use.

## Project Structure

The project is structured into the following packages:

- **MNIST.domain**: Contains classes related to the MNIST dataset and image processing.
- **MNIST.reader**: Handles reading the MNIST dataset from files.
- **neural_network.domain**: Contains the core neural network components, including layers and neurons.
- **neural_network.service**: Provides services for training, testing, and managing the neural network.
- **neural_network.reader**: Handles saving and loading the trained neural network model.
- **neural_network.exception**: Contains custom exceptions related to the neural network operations.
- **neural_network.interfaces**: Defines interfaces for serializable functions used in the network.

## Getting Started

### Prerequisites


- **Java**: Make sure you have Java Development Kit (JDK) installed on your system.
- **MNIST Dataset**: Download the MNIST dataset files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, and `t10k-labels.idx1-ubyte`) from [DATA](https://drive.google.com/drive/folders/1WznmCA4huRWaZMk5ubp4xsPGcECDLS7u?usp=sharing) and place them in the `data` directory.

### Running the Application

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/mnist-neural-network.git
    cd mnist-neural-network
    ```

2. **Compile the Code:**

    ```bash
    javac -d bin src/**/*.java
    ```

3. **Run the Application:**

    ```bash
    java -cp bin NeuralNetworkApp
    ```

## Menu Options

When you run the application, you will be presented with the following menu:

1. **Train Neural Network for MNIST**: Train the neural network on the MNIST training dataset.
2. **Test Neural Network for MNIST**: Test the trained neural network on the MNIST test dataset.
3. **Test Neural Network for Image**: Test the network on a custom image of a handwritten digit.
0. **Exit Program**: Exit the application.

### Training the Network

When you choose to train the network, the program will read the MNIST training data and train the network using backpropagation. The learning rate is halved after each epoch to help the network converge.

### Testing the Network

You can test the network on both the MNIST test dataset and custom images. The network's accuracy on the test dataset will be displayed, and the result of the custom image test will be shown along with the expected and actual outputs.

### Saving and Loading the Network

The trained network is automatically saved after training, allowing you to load it later for testing without retraining.

### Custom Image Testing

To test the network on a custom image:

1. Place the image in the `Images` directory.
2. The image should be a grayscale image of size 28x28 pixels. If it is not, the program will resize it automatically.
3. Choose option `[3] Test Neural Network for Image` in the menu, enter the expected digit value, and the image name.

## Example Usage

```
Enter Choice: 1
# Trains the neural network

Enter Choice: 2
# Tests the network on the MNIST test dataset and displays accuracy

Enter Choice: 3
Enter Digit Value: 7
Enter Image Name: digit7.png
# Tests the network on a custom image of the digit 7
```
