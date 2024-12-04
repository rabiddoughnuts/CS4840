from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Ensure reproducibility
tf.random.set_seed(42)

# Load and preprocess the data (assuming CIFAR-10 dataset)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert to grayscale
X_train = np.mean(X_train, axis=-1, keepdims=True)
X_test = np.mean(X_test, axis=-1, keepdims=True)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Verify the shapes of the data
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Class names for CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                 "dog", "frog", "horse", "ship", "truck"]

# Plot the data
def plot_digits(instances, labels, images_per_row=5):
    labels = labels.flatten()  # Flatten the labels array
    fig, ax = plt.subplots(2, 5, figsize=(8, 4))
    for i in range(len(instances)):
        idx = i // images_per_row
        idy = i % images_per_row 
        ax[idx, idy].imshow(instances[i].squeeze(), cmap="gray")
        ax[idx, idy].set_title(class_names[labels[i]])
        ax[idx, idy].axis("off")
    plt.show()

example_images = X_train[:10]
example_labels = y_train[:10]
plot_digits(example_images, example_labels, images_per_row=5)

# Train a Multi-Layer Perceptron
drop_rates = [0.0, 0.3, 0.5]  # Please use 0.0, 0.3, and 0.5, respectively

def answer_one(drop_rate):
    print("Starting answer_one function")
    try:
        print("Adding input layer")
        model = keras.models.Sequential()
        input_shape = (32, 32, 1)
        print(f"Input shape: {input_shape}")
        model.add(keras.layers.InputLayer(shape=input_shape))  # Input layer with shape [32, 32, 1]
        print("Input layer added")
        
        print("Adding flatten layer")
        model.add(keras.layers.Flatten())  # Flatten layer
        print("Flatten layer added")
        
        print("Adding first dense layer")
        model.add(keras.layers.Dense(300, activation="relu"))  # Dense hidden layer, 300 neurons, ReLU
        print("First dense layer added")
        
        print("Adding first dropout layer")
        model.add(keras.layers.Dropout(drop_rate))  # Dropout layer using drop_rate to address overfitting
        print("First dropout layer added")
        
        print("Adding second dense layer")
        model.add(keras.layers.Dense(100, activation="relu"))  # Dense hidden layer, 100 neurons, ReLU
        print("Second dense layer added")
        
        print("Adding second dropout layer")
        model.add(keras.layers.Dropout(drop_rate))  # Dropout layer using drop_rate to address overfitting
        print("Second dropout layer added")
        
        print("Adding output layer")
        model.add(keras.layers.Dense(10, activation="softmax"))  # Dense output layer, 10 neurons, softmax
        print("Output layer added")
        
        print("Model created")
        
        model.compile(
            loss="sparse_categorical_crossentropy",  # Loss function
            optimizer="adam",  # Optimization algorithm: adam
            metrics=["accuracy"]  # Evaluation metrics: accuracy
        )
        
        print("Model compiled")
        
        # Add print statements to check the data before training
        print(f"Training with drop_rate: {drop_rate}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        model.fit(
            X_train, y_train,  # Use X_train, y_train for training
            epochs=20,  # epochs, 20
            batch_size=32,  # batch size, 32
            validation_data=(X_test, y_test)  # Use X_test, y_test for validation
        )
        
        print("Model training completed")
        
        y_proba = model.predict(X_test)  # Calculate prediction probabilities on X_test
        y_pred = np.argmax(y_proba, axis=1)  # Obtain the predicted classes from y_proba
        
        print("Prediction completed")
        
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        microf1 = f1_score(y_test, y_pred, average='micro')  # Calculate micro f1
        macrof1 = f1_score(y_test, y_pred, average='macro')  # Calculate macro f1
        
        print("Metrics calculated")
        
        return accuracy, microf1, macrof1
    except Exception as e:
        print(f"An error occurred: {e}")

# Run your function for each drop rate and print the results
for drop_rate in drop_rates:
    print(f"Running answer_one with drop_rate: {drop_rate}")
    accuracy, microf1, macrof1 = answer_one(drop_rate)
    print(f"Dropout Rate: {drop_rate}")
    print(f"Accuracy: {accuracy}, Micro F1 Score: {microf1}, Macro F1 Score: {macrof1}\n")