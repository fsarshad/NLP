from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_model_from_name(model_name): 
    """
    Load a trained model based on the given model name.

    This function maps the model name to its corresponding file path and loads the model using Keras.

    Parameters:
    - model_name (str): The name of the model to be loaded. Should be one of the keys in the `model_name_path_map` dictionary.

    Returns:
    - model (keras.Model): The loaded Keras model object.

    Raises:
    - KeyError: If the provided `model_name` is not found in the `model_name_path_map` dictionary.
    """
    model_name_path_map = {
        'Conv1D': 'models/model_conv1d.h5',
        'GloVe100D': 'models/model_glove.h5',
        'BERT': 'models/model_BERT.h5',
        'Universal Sentence Encoder (USE)': 'models/model_USE.h5',
        'GPT-2': 'models/model_GPT2.h5'
    }
    model_path = model_name_path_map[model_name]
    model = load_model(model_path)
    return model

def predict_sentiments(model, data):
    """
    Predict the sentiment labels for the given data using the provided model.

    This function uses the trained model to make predictions on the input data and returns the predicted sentiment labels.

    Parameters:
    - model (keras.Model): The trained Keras model used for making predictions.
    - data (numpy.ndarray): The input data to be predicted. Should be a 2D numpy array of shape (num_samples, max_length).

    Returns:
    - y_pred_idx (numpy.ndarray): The predicted sentiment labels as a 1D numpy array of shape (num_samples,).
      Each element represents the index of the predicted sentiment class.
    """
    # Make predictions using the model
    y_pred = model.predict(data)
    y_pred_idx = np.argmax(y_pred, axis=1)
    return y_pred_idx

def calculate_accuracy(y_pred, y_true):
    """
    Calculate the accuracy of the predicted sentiment labels compared to the true labels.

    This function computes the accuracy by comparing the predicted labels with the true labels.

    Parameters:
    - y_pred (numpy.ndarray): The predicted sentiment labels as a 1D numpy array of shape (num_samples,).
    - y_true (numpy.ndarray): The true sentiment labels as a 1D numpy array of shape (num_samples,).

    Returns:
    - accuracy (float): The calculated accuracy value between 0 and 1.
    """
    return np.mean(y_pred == y_true)

def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    Plot the confusion matrix for the provided true and predicted sentiment labels.

    This function generates a heatmap visualization of the confusion matrix using Seaborn.

    Parameters:
    - y_true (numpy.ndarray): The true sentiment labels as a 1D numpy array of shape (num_samples,).
    - y_pred (numpy.ndarray): The predicted sentiment labels as a 1D numpy array of shape (num_samples,).
    - model_name (str, optional): The name of the model to be displayed in the plot title. Default is 'Model'.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

