import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog, messagebox
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights,ResNet101_Weights
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from activation_functions import  ReLU, Dense, Tanh, Sigmoid
from NN import NeuralNetwork


class MiniMatlab:
    def __init__(self, root):

        self.root = root
        self.root.title("Mini MATLAB")
        self.root.geometry("400x400")

        # Initialize variables
        self.data_type_var = tk.StringVar()
        self.data_type_var.set("Select Data Type")

        self.problem_type_var = tk.StringVar()
        self.num_classes_var = tk.StringVar()

        self.processed_var = tk.BooleanVar()
        self.not_processed_var = tk.BooleanVar()

        # Create dropdown menu for selecting data type
        self.data_type_dropdown = tk.OptionMenu(root, self.data_type_var, "Select Data Type", "CSV Data", "Images")
        self.data_type_dropdown.pack(pady=20)

        # Create checkboxes for data processing options
        self.processed_checkbox = tk.Checkbutton(root, text="Processed", variable=self.processed_var, onvalue=True,
                                                 offvalue=False)
        self.not_processed_checkbox = tk.Checkbutton(root, text="Not Processed", variable=self.not_processed_var,
                                                     onvalue=True, offvalue=False)

        self.processed_checkbox.pack(pady=10)
        self.not_processed_checkbox.pack(pady=10)

        # Create buttons
        self.load_data_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.next_button = tk.Button(root, text="Next", state=tk.DISABLED, command=self.open_second_window)

        # Place buttons on the window
        self.load_data_button.pack(pady=20)
        self.next_button.pack(pady=20)

        # Initialize variables to store model and training window
        self.input_dim = None
        self.hidden_layers = None
        self.neurons = None
        self.learning_rate = None
        self.activation = None
        self.epochs = None
        self.goal = None
        self.model = None
        self.training_window = None
        self.problem_type = None
        self.cnn_type = None

    def validate_image_folder(self, folder_path):
        # Check if the folder contains at least two subfolders
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        return len(subfolders) >= 2

    def load_data(self):
        data_type = self.data_type_var.get()

        if data_type == "Select Data Type":
            messagebox.showerror("Error", "Please select a valid data type.")
            return

        if data_type == "CSV Data":
            file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if file_path and file_path.lower().endswith('.csv'):
                print(f"CSV file loaded: {file_path}")
                processed = self.processed_var.get()
                not_processed = self.not_processed_var.get()

                if processed and not not_processed:
                    success = self.load_csv_file(file_path, True)
                    if not success:
                        print("Sth wrong with the csv file")

                elif not processed and not_processed:
                    success = self.load_csv_file(file_path, False)
                    if not success:
                        print("Sth wrong with the csv file")
                else:
                    messagebox.showerror("Error", "Please select one of the data processing options.")
                    return

                self.enable_next_button()
        elif data_type == "Images":
            folder_path = filedialog.askdirectory()
            if folder_path and self.validate_image_folder(folder_path):
                print(f"Image folder loaded: {folder_path}")
                self.load_images_folder(folder_path)
                self.enable_next_button()
            else:
                messagebox.showerror("Error", "Please insert a folder with at least 2 classes.")
                return

    def detect_num_classes_from_folder(self, folder_path):
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        if len(subfolders) == 2:
            self.num_classes_var.set('2')
        elif len(subfolders) > 2:
            self.num_classes_var.set(f'{len(subfolders)}')

    def preprocess_dataset(self):
        print("LATERRRRRR")
        pass

    def enable_next_button(self):
        self.next_button.config(state=tk.NORMAL)

    def create_classification_model(self):
        self.model = NeuralNetwork(self.activation)
        self.model.add_layer(Dense(self.input_dim, self.neurons, self.learning_rate))
        if self.activation == 'relu':
            self.model.add_layer(ReLU())
            for num in range(self.hidden_layers):
                if self.neurons == 1:
                    break
                self.model.add_layer(Dense(self.neurons, self.neurons - 2, self.learning_rate))
                self.model.add_layer(ReLU())
                self.neurons = self.neurons - 2
        elif self.activation == 'tanh':
            self.model.add_layer(Tanh())
            for num in range(self.hidden_layers):
                if self.neurons == 1:
                    break
                self.model.add_layer(Dense(self.neurons, self.neurons - 2, self.learning_rate))
                self.model.add_layer(Tanh())
                self.neurons = self.neurons - 2
        else:
            self.model.add_layer(Sigmoid())
            for num in range(self.hidden_layers):
                if self.neurons == 1:
                    break
                self.model.add_layer(Dense(self.neurons, self.neurons - 2, self.learning_rate))
                self.model.add_layer(Sigmoid())
                self.neurons = self.neurons - 2

        self.model.add_layer(Dense(self.neurons, self.num_classes_var))


    def open_second_window(self):
        self.root.withdraw()

        # Create a new window for the second interface
        second_window = tk.Toplevel()
        second_window.title("Second Window")
        second_window.geometry("400x800")

        # Check if the second window still exists
        if not second_window.winfo_exists():
            messagebox.showerror("Error",
                                 "Couldn't present window.")
            return

        # Add labels and entry fields for user input
        if self.data_type_var.get() == "Images":
            model_cnn_type_label = tk.Label(second_window, text="Which CNN model Would you like to choose?")
            model_cnn_type_options = ["resent18", "resent50", "resnet101"]
            model_cnn_var = tk.StringVar()
            model_cnn_dropdown = tk.OptionMenu(second_window, model_cnn_var, *model_cnn_type_options)
            model_cnn_var.set(model_cnn_type_options[0])
            model_cnn_type_label.pack(pady=10)
            model_cnn_dropdown.pack(pady=10)

        else:
            hiddens_label = tk.Label(second_window, text="Number of Hidden Layers:")
            hiddens_entry = tk.Entry(second_window)
            hiddens_label.pack(pady=10)
            hiddens_entry.pack(pady=10)

            neurons_label = tk.Label(second_window, text="Number of Neurons of first Hidden Layer:")
            neurons_entry = tk.Entry(second_window)
            neurons_label.pack(pady=10)
            neurons_entry.pack(pady=10)

            problem_label = tk.Label(second_window,
                                     text="Please type 0 if the problem is classification and 1 if regression:")
            problem_entry = tk.Entry(second_window)
            problem_label.pack(pady=10)
            problem_entry.pack(pady=10)

            activation_label = tk.Label(second_window, text="Activation Function:")
            activation_options = ["relu", "sigmoid", "tanh"]
            activation_var = tk.StringVar()
            activation_dropdown = tk.OptionMenu(second_window, activation_var, *activation_options)
            activation_var.set(activation_options[0])

            # Add input fields and buttons to the window
            activation_label.pack(pady=10)
            activation_dropdown.pack(pady=10)

        learning_rate_label = tk.Label(second_window, text="Learning Rate (0-1):")
        learning_rate_entry = tk.Entry(second_window)

        epochs_label = tk.Label(second_window, text="Maximum Number of Epochs (1-1000):")
        epochs_entry = tk.Entry(second_window)

        goal_label = tk.Label(second_window, text="Accuracy Goal (0-99):")
        goal_entry = tk.Entry(second_window)

        learning_rate_label.pack(pady=10)
        learning_rate_entry.pack(pady=10)

        epochs_label.pack(pady=10)
        epochs_entry.pack(pady=10)

        goal_label.pack(pady=10)
        goal_entry.pack(pady=10)

        # Function to validate if the input is a valid number
        def validate_number(entry_text, max_value=float('inf')):
            try:
                value = float(entry_text)
                if not (0 <= value <= max_value):
                    raise ValueError
                return True
            except ValueError:
                return False

        # Function to handle the OK button click
        def on_ok_click():
            # Validate input values
            if not self.data_type_var.get() == "Images":
                self.problem_type = int(problem_entry.get())
                self.hidden_layers = int(hiddens_entry.get())
                self.neurons = int(neurons_entry.get())
            else:
                self.cnn_type = model_cnn_var.get()
                self.activation = activation_var.get()

            self.learning_rate = float(learning_rate_entry.get())
            self.epochs = int(epochs_entry.get())
            self.goal = float(goal_entry.get())

            if self.data_type_var.get() == "Images":
                if not (validate_number(self.learning_rate, max_value=1) and
                        validate_number(self.epochs, max_value=1000) and
                        validate_number(self.goal, max_value=99)):
                    messagebox.showerror("Error",
                                         "Invalid input. Please enter valid numeric values within the specified range.")
                    return
            else:
                if not (validate_number(self.hidden_layers, max_value=10) and
                        validate_number(self.neurons, max_value=100) and
                        validate_number(self.problem_type, max_value=1) and
                        validate_number(self.learning_rate, max_value=1) and
                        validate_number(self.epochs, max_value=1000) and
                        validate_number(self.goal, max_value=99)):
                    messagebox.showerror("Error",
                                         "Invalid input. Please enter valid numeric values within the specified range.")
                    return

                if self.hidden_layers > self.input_dim:
                    messagebox.showerror("Error",
                                         "Invalid input. Please enter number of hidden layers less than number of features!")
                    return

                self.create_classification_model()

            # Hide the second window
            second_window.withdraw()
            self.open_third_window()

        ok_button = tk.Button(second_window, text="OK", command=on_ok_click)
        ok_button.pack(pady=20)

        # Add a button to close the program
        exit_button = tk.Button(second_window, text="Exit", command=second_window.destroy)
        exit_button.pack(pady=20)

        # Function to load classification model with appropriate loss function
        def load_cnn_classification_model():
            if self.cnn_type == 'resent18':
                self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif self.cnn_type == 'resent50':
                self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

        # Load Images data
        if self.data_type_var.get() == "Images":
            load_cnn_classification_model()

    def open_third_window(self):
        # Create a new window for the training interface
        self.training_window = tk.Toplevel()
        self.training_window.title("Training Window")
        self.training_window.geometry("800x800")
        self.training_window.resizable(width=True, height=True)

        # Add label for dataset Summary
        data_label = tk.Label(self.training_window, text="The dataset has been split as a following:")
        data_label.pack(pady=10)

        data_summary_text = self.get_data_summary_text()
        data_summary_label = tk.Label(self.training_window, text=data_summary_text, justify=tk.LEFT)
        data_summary_label.pack(pady=10)

        # Add label and image to display the created model
        model_label = tk.Label(self.training_window,
                               text="*************************************************************")
        model_label.pack(pady=10)

        # Display the model summary
        model_summary_text = self.get_model_summary_text()
        model_summary_label = tk.Label(self.training_window, text=model_summary_text, justify=tk.LEFT)
        model_summary_label.pack(pady=10)

        def on_train_click():
            self.training_window.withdraw()
            self.open_forth_window()

        # Add a button to start training
        self.progress_text = tk.StringVar()
        self.progress_text.set("Training Progress")
        train_button = tk.Button(self.training_window, text="Train",
                                 command=on_train_click)
        train_button.pack(pady=20)

        exit_button = tk.Button(self.training_window, text="Exit", command=self.training_window.destroy)
        exit_button.pack()

    def open_forth_window(self):
        self.trainingPrcoess_window = tk.Toplevel()
        self.trainingPrcoess_window.title("Training Process Window")
        self.trainingPrcoess_window.geometry("800x400")

        # Create a Canvas widget with a vertical scrollbar
        canvas = tk.Canvas(self.trainingPrcoess_window)
        scrollbar = ttk.Scrollbar(self.trainingPrcoess_window, orient="vertical", command=canvas.yview)
        canvas.config(yscrollcommand=scrollbar.set)

        # Create a Frame inside the Canvas to hold the content
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor=tk.NW)

        # Add label for dataset Summary
        label = tk.Label(self.training_window, text="The Training Prcoess:")
        label.pack(pady=10)

        label = tk.Label(self.training_window, text="=== * 60")
        label.pack(pady=10)

        self.progress_text = tk.StringVar()
        self.progress_text.set("Training Progress")

        training_summary_text = []

        if self.problem_type == 0:
            if self.num_classes_var == 2:
                # self.model.train( self.X_train, self.Y_train, self.X_val, self.Y_val,self.progress_text,self.trainingPrcoess_window, self.epochs, self.learning_rate,self.goal, loss_function='binary_crossentropy')
                losses = self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val, epochs=self.epochs,
                                          batch_size=1 if self.total_number_samples <= 10 else 4,
                                          accuracy_goal=self.goal, loss_function='cross_entropy',
                                          print_fn=lambda x: training_summary_text.append(x))
            else:
                losses = self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val, epochs=self.epochs,
                                          batch_size=1 if self.total_number_samples <= 10 else 4,
                                          accuracy_goal=self.goal, loss_function='categorical_cross_entropy',
                                          print_fn=lambda x: training_summary_text.append(x))
        else:
            self.num_classes_var = 1
            losses = self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val, epochs=self.epochs,
                                      batch_size=1 if self.total_number_samples <= 10 else 4, accuracy_goal=self.goal,
                                      loss_function='mse', print_fn=lambda x: training_summary_text.append(x))

        def train_cnn_model(epochs, goal):
            batch_size = 32
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            num_classes = len(self.train_dataset.classes)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

            # Set the model to training mode
            self.model.train()

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            # Train the model
            num_epochs = 10
            losses = []
            for epoch in range(num_epochs):
                running_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    losses.append(loss.item())

                average_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

        if self.data_type_var.get() == "Images":
            losses = train_cnn_model(self.epochs, self.goal)

        # training_summary_label = tk.Label(self.trainingPrcoess_window, text="\n".join(training_summary_text),
        #                                   justify=tk.LEFT)
        # training_summary_label.pack(pady=10)

        training_summary_text = "\n".join(training_summary_text)
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=30)
        text_widget.insert(tk.END, training_summary_text)
        text_widget.pack(padx=10, pady=10)

        # Create a new window for the plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Training Plot")

        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, linestyle='-', marker='')

        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Over Epochs')

        # Create a canvas for the plot
        canvas_plot = FigureCanvasTkAgg(fig, master=plot_window)
        canvas_widget = canvas_plot.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Draw the updated plot
        canvas_plot.draw()

        self.show_validation_button()

        # Add an event binding to update the scroll region when the Canvas is resized
        def on_canvas_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", on_canvas_configure)

        # Pack the Canvas and Scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Start the Tkinter event loop
        self.trainingPrcoess_window.mainloop()

    def load_images_folder(self,folder_path):
        # Define a transformation for the input image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the ImageFolder dataset
        dataset = ImageFolder(folder_path, transform=transform)

        # Create a DataLoader for batching
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])


    def load_csv_file(self, file_path, processed):
        # Add your preprocessing logic here based on the CSV file
        # This is just a placeholder, replace it with your actual preprocessing steps
        df = pd.read_csv(file_path)
        processed_data = df  # Placeholder, you should perform actual preprocessing
        self.input_dim = len(processed_data.columns) - 1

        if processed:

            # Step 2: Check for string values in each column
            for column in df.columns:
                if df[column].dtype == 'O':  # 'O' stands for object, which typically represents strings
                    print(f"Column '{column}' contains string values.")
                    messagebox.showerror("Error",
                                         f"Column '{column}' contains string values..")
                    return

            # Step 3: Separate X and Y
            X = df.iloc[:, :-1]  # All columns except the last one
            Y = df.iloc[:, -1]  # Last column

            self.total_number_samples = len(X)

            # Step 4: Normalize X values
            scaler = StandardScaler()
            X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            Y_encoded = Y.values
            # Step 5: Convert categorical labels in Y to numerical values only if Y is not already integer
            if Y.dtype != 'int64':
                label_encoder = LabelEncoder()
                Y_encoded = label_encoder.fit_transform(Y)

            # Step 6: Get the number of classes in Y
            self.num_classes_var = len(set(Y_encoded))

            # Step 8: Display the processed data
            print("X_normalized:")
            print(X_normalized.head())
            print("\nY_encoded:")
            print(Y_encoded)
            print(f"\nNumber of classes in Y: {self.num_classes_var}")

            # Step 9: Split the data into training, validation, and testing sets
            self.X_train, X_temp, self.Y_train, Y_temp = train_test_split(X_normalized, Y_encoded, test_size=0.2,
                                                                          random_state=42)
            self.X_val, self.X_test, self.Y_val, self.Y_test = train_test_split(X_temp, Y_temp, test_size=0.5,
                                                                                random_state=42)
            self.X_train = self.X_train.values
            self.X_val = self.X_val.values
            self.X_test = self.X_test.values
            self.Y_val = np.eye(self.num_classes_var)[self.Y_val]
            self.Y_train = np.eye(self.num_classes_var)[self.Y_train]
            self.Y_test = np.eye(self.num_classes_var)[self.Y_test]

            # Step 10: Display the shapes of the resulting sets
            print(f"X_train shape: {self.X_train.shape}, Y_train shape: {self.Y_train.shape}")
            print(f"X_val shape: {self.X_val.shape}, Y_val shape: {self.Y_val.shape}")
            print(f"X_test shape: {self.X_test.shape}, Y_test shape: {self.Y_test.shape}")
            return True
        else:
            print("CSV file not processed we will deal with this later")
            self.preprocess_dataset()
            return False

    def get_model_summary_text(self):
        # Capture the model summary in a string
        if self.data_type_var.get() == "Images":
            return f'{self.model}'
        else:
            summary_text = []
            summary_text.append("Neural Network Model Summary:")
            model_summary = []
            self.model.model_summary(print_fn=lambda x: model_summary.append(x))
            summary_text.extend(model_summary[1:])  # Skip the first line as it contains the model name and type
            return "\n".join(summary_text)

    def get_data_summary_text(self):
        # Capture the model summary in a string
        if self.data_type_var.get() == "Images":
            summary_text = [f"Size of the training data is {len(self.train_dataset)}",
                            f"Size of the testing data is {len(self.val_dataset)}",
                            f"Number of classes {self.num_classes_var}"]
        else:
            summary_text = [f"Number of classes we have {self.num_classes_var} ",
                            f"total samples we have {self.total_number_samples}",
                            f"X_train shape: {self.X_train.shape}, Y_train shape: {self.Y_train.shape}",
                            f"X_val shape: {self.X_val.shape}, Y_val shape: {self.Y_val.shape}",
                            f"X_test shape: {self.X_test.shape}, Y_test shape: {self.Y_test.shape}"]
        return "\n".join(summary_text)

    def show_validation_button(self):
        # Add a button to perform validation
        test_button = tk.Button(self.trainingPrcoess_window, text="Test", command=self.validate_model)
        exit_button = tk.Button(self.trainingPrcoess_window, text="Exit", command=self.trainingPrcoess_window.destroy)
        test_button.pack(side=tk.LEFT, padx=20)
        exit_button.pack(side=tk.LEFT, padx=20)



    def validate_model(self):
        # Dummy validation process, replace with your actual validation process
        # messagebox.showinfo("Validation", "Validation completed successfully!")
        # Testing the model
        accuracy, loss, confusion_mat, precision, recall, f1 = self.model.test(self.X_test, self.Y_test)
        self.trainingPrcoess_window.withdraw()

        self.test_window = tk.Toplevel()
        self.test_window.title("Testing Results")
        self.test_window.geometry("800x400")

        results = ["Test Accuracy: {:.4f}".format(accuracy), "Test loss: {:.4f}".format(loss),
                   f"Confusion Matrix:\n{confusion_mat}",
                   f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-score: {f1:.4f}"]
        # Add label for dataset Summary
        label = tk.Label(self.test_window, text="The Accuracy and other matrics results over testing data:")
        label.pack(pady=10)

        for r in results:
            # Add label for dataset Summary
            label = tk.Label(self.test_window, text=r)
            label.pack(pady=10)

        exit_button = tk.Button(self.test_window, text="Exit", command=self.test_window.destroy)
        exit_button.pack(pady=20)
