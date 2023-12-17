import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from activation_functions import grad_softmax_crossentropy_with_logits,cross_entropy_loss,categorical_cross_entropy_loss,mean_squared_error_loss

np.random.seed(42)

class NeuralNetwork:
    def __init__(self, activation_type):
        self.layers = []
        self.activation_type = activation_type

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        """ Perform an forward pass through all the layers of the network """

        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output, y_true):
        """ Perform backward pass given the output and true labels """
        # Assuming the last layer is a Dense layer with softmax and cross-entropy
        loss_grad = grad_softmax_crossentropy_with_logits(output, y_true)

        for layer in reversed(self.layers):
            loss_grad = layer.backward(layer.input, loss_grad)


    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32, accuracy_goal=95,
              loss_function='categorical_cross_entropy', print_fn=print):

        print_fn("The Training Process has begun")
        print_fn("==============================")
        if not self.layers:
            print_fn("Please add layers to the network before training.")
            raise ValueError("Please add layers to the network before training.")
        losses = []
        for epoch in range(epochs):
            # Training phase
            total_loss = 0
            correct_predictions = 0
            train_loader = NumpyDataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
            for batch in train_loader:
                x_batch, y_batch = batch
                # Forward pass
                output = self.forward(x_batch)

                # Compute loss
                if loss_function == 'cross_entropy':
                    loss = cross_entropy_loss(output, y_batch)
                elif loss_function == 'categorical_cross_entropy':
                    loss = categorical_cross_entropy_loss(output, y_batch)
                else:
                    loss = mean_squared_error_loss(output, y_batch)

                total_loss += loss
                losses.append(loss)

                # Backward pass
                self.backward(output, y_batch)
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(predictions == true_labels)

            # Average loss and accuracy for the epoch
            average_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / len(x_train)
            print_fn(f"Epoch {epoch + 1}/{epochs} (Training) - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Validation phase
            val_output = self.forward(x_val)
            if loss_function == 'cross_entropy':
                val_loss = cross_entropy_loss(val_output, y_val)
            elif loss_function == 'categorical_cross_entropy':
                val_loss = categorical_cross_entropy_loss(val_output, y_val)
            else:
                val_loss = mean_squared_error_loss(val_output, y_val)

            val_predictions = np.argmax(val_output, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)
            val_accuracy = np.sum(val_predictions == val_true_labels) / len(x_val)

            print_fn(f"Epoch {epoch + 1}/{epochs} (Validation) - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print_fn("===============================================================================")

            # Check for early stopping
            if accuracy >= (accuracy_goal / 100):
                print_fn("Training stopped early. Reached accuracy goal.")
                print("Training stopped early. Reached accuracy goal.")
                break

        print("Training complete")
        print_fn("Training complete")

        return losses


    def test(self, x_test, y_test):
        if not self.layers:
            raise ValueError("Please add layers to the network before testing.")

        # Forward pass on the test data
        test_logits = self.forward(x_test)

        # Assuming the last layer produces logits for a softmax function
        # Calculate softmax probabilities
        exp_logits = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
        softmax_outputs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Convert one-hot encoded y_test to class labels if it's one-hot encoded
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test

        # Calculate the predicted class labels
        predictions = np.argmax(softmax_outputs, axis=1)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_test_labels)
        print("Test Accuracy: {:.4f}".format(accuracy))

        # Optional: Calculate the loss (e.g., cross-entropy)
        loss = cross_entropy_loss(test_logits, y_test)
        print("Test Loss: {:.4f}".format(loss))

        confusion_mat = confusion_matrix(y_test_labels, predictions)
        precision = precision_score(y_test_labels, predictions)
        recall = recall_score(y_test_labels, predictions)
        f1 = f1_score(y_test_labels, predictions)

        print(f"Confusion Matrix:\n{confusion_mat}")
        print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-score: {f1:.4f}")

        return accuracy, loss , confusion_mat, precision, recall, f1

    def model_summary(self,print_fn):
        # Print a summary of the model architecture
        print_fn("\n=== Model Summary ===")
        print("\n=== Model Summary ===")
        print_fn("{:<15} {:<15} {:<15} {}".format("Layer", "Units", "Activation", "Params"))
        print("{:<15} {:<15} {:<15} {}".format("Layer", "Units", "Activation", "Params"))
        print_fn("=" * 45)
        print("=" * 45)
        total_params = 0
        for i in range(0,len(self.layers), 2):
            layer_type = "Input" if i == 0 else ("Hidden" if i < len(self.layers) - 1 else "Output")
            units = self.layers[i]
            activation = self.activation_type if i == 0 else (self.activation_type if i < len(self.layers) - 1 else "Softmax")
            params = len(units.weights) * (len(self.layers[i].weights) + 1)  # +1 for biases
            total_params += params
            print_fn("{:<15} \u2003 {} \u2003 {} \u2003 {:<15} ".format(layer_type, units.weights.shape, activation, params))
            print("{:<15} \u2003{} \u2003 {}  \u2003 {:<15} ".format(layer_type, units.weights.shape, activation, params))
        print_fn("=" * 45)
        print("=" * 45)
        print_fn("Total Parameters: {}".format(total_params))
        print("Total Parameters: {}".format(total_params))
        print_fn("===================")
        print("===================")


class NumpyDataLoader:
    def __init__(self, x, y, batch_size=1, shuffle=False):
        self.dataset = x
        self.labels = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

    def __iter__(self):
        # Shuffle the indices if necessary
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if len(self.indices) == 0:
            raise StopIteration

        # Select indices for the next batch
        selected_indices = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]

        # Extract the batch from the dataset
        batch = [self.dataset[i] for i in selected_indices]
        selected_labels = [self.labels[i] for i in selected_indices]
        return np.array(batch), np.array(selected_labels)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size




