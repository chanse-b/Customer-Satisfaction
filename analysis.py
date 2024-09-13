import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from sklearn.linear_model import LogisticRegression
from data_processor import getData, preprocess_data
from plotter import*


# Function to build the neural network model
def build_model(input_shape):
    """
    based off of coursera  Neural Networks for Handwritten Digit Recognition, Binary
    """
    model = Sequential([
        Dense(25, activation='sigmoid', input_shape=(input_shape,)),   # sigmoid originally
        Dense(15, activation='sigmoid'),
        Dense(1, activation='sigmoid'),  # Sigmoid activation for binary classification
    ])

    model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=['accuracy'],
    )
    return model

def trainer(use_neural_network=True):
    train_df, test_df = getData()
    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)

    # Convert data to float32, errors when processing
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    if use_neural_network:
        # Use Neural Network
        
        model = build_model(X_train.shape[1])
        # Train the Neural Network
        history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2) # epochs=2 if you want to test quickly
        # Evaluate the Neural Network
        train_loss, train_accuracy = model.evaluate(X_train, y_train)
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        weights = [layer.get_weights()[0] for layer in model.layers if isinstance(layer, Dense)]
        # Plot the weights as a heatmap
        plot_dense_layer_weights(weights, X_train.columns)
        y_unadjusted = model.predict(X_test)
    else:
        # Use Logistic Regression
        model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warning appears
        model.fit(X_train, y_train)
        # Evaluate Logistic Regression
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        plot_model_coefficients(model, X_train.columns)
        y_unadjusted = model.predict_proba(X_test)[::,1]
    y= np.where(y_unadjusted >= .5, 1, 0)
   
    class_labels = ['neutral or dissatisfied', 'satisfied']
    plot_confusion_matrix(y_test, y, class_labels)
    generateROC(y_test, y_unadjusted)


if __name__ == "__main__":
    trainer(False)
