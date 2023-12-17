import numpy as np


np.random.seed(42)


class Layer:
    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        self.weights = np.zeros(shape=(input.shape[1], 10))
        self.bias = np.zeros(shape=(10,))
        pass

    def forward(self, input):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, 10]
        """
        self.input = input  # Store the input to the layer
        output = np.matmul(input, self.weights) + self.bias
        return output


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate

        # initialize weights with small random numbers. We use normal initialization
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        self.input = input  # Store the input to the layer
        return np.matmul(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output), input))
        grad_biases = np.sum(grad_output, axis=0)

        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input


class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass

    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        self.input = input  # Store the input to the layer
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output * relu_grad

class Tanh(Layer):
    def __init__(self):
        """Tanh layer applies elementwise hyperbolic tangent to all inputs"""
        pass

    def forward(self, input):
        """Apply elementwise hyperbolic tangent to [batch, input_units] matrix"""
        self.input = input  # Store the input to the layer
        return np.tanh(input)

    def backward(self,input, grad_output):
        """Compute gradient of loss w.r.t. tanh input"""
        tanh_grad = 1.0 - np.tanh(input)**2
        return grad_output * tanh_grad

class Sigmoid(Layer):
    def __init__(self):
        """Sigmoid layer applies elementwise hyperbolic tangent to all inputs"""
        pass

    def forward(self, input):
        self.input = input  # Store the input to the layer
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        sigmoid_output = self.forward(input)
        return grad_output * sigmoid_output * (1 - sigmoid_output)



def softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, y_true_one_hot):
    """
    Compute crossentropy gradient from logits[batch,n_classes] and one-hot encoded true labels.
    """
    # Calculate softmax
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    # Compute the gradient
    # The gradient is the difference between softmax outputs and one-hot true labels
    # divided by the number of samples in the batch for averaging
    grad = (softmax - y_true_one_hot) / logits.shape[0]
    return grad

def mean_squared_error_loss(predictions, y_true_one_hot):
    """
    Compute mean squared error between predictions and true one-hot encoded targets.

    Parameters:
    - predictions: The predicted values from the model.
    - y_true_one_hot: The true target values in one-hot encoded format.

    Returns:
    - The mean squared error loss.
    """
    # Ensure both predictions and targets are numpy arrays
    predictions = np.array(predictions)

    # Compute the mean squared error
    mse = np.mean((predictions - y_true_one_hot)**2)

    return mse



def cross_entropy_loss(logits, y_true_one_hot):
    """Compute crossentropy from logits and one-hot encoded true labels"""
    # Apply softmax to logits
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    softmax_outputs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Compute the cross-entropy
    cross_entropy = -np.sum(y_true_one_hot * np.log(softmax_outputs + 1e-15), axis=1)  # 1e-15 for numerical stability
    return np.mean(cross_entropy)


def categorical_cross_entropy_loss(logits, y_true_one_hot):
    """
    Compute categorical cross entropy from logits and one-hot encoded true labels.

    Parameters:
    - logits: The raw, unnormalized predictions (logits) from the model.
    - y_true_one_hot: The true labels in one-hot encoded format.

    Returns:
    - The mean categorical cross entropy loss.
    """

    # Apply softmax to logits
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    softmax_outputs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Compute the categorical cross entropy
    cross_entropy = -np.sum(y_true_one_hot * np.log(softmax_outputs + 1e-15), axis=1)  # 1e-15 for numerical stability

    # Calculate the mean loss
    mean_loss = np.mean(cross_entropy)

    return mean_loss
