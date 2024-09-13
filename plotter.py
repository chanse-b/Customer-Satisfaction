import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_dense_layer_weights(weights, feature_names):
    """
    Plots heat map of the weights for each Dense layer in the model.

    Parameters:
    - weights: List of numpy arrays containing weights for each Dense layer.
    - feature_names: List of names for the input features.
    """
    for i, layer_weights in enumerate(weights):
        plt.figure(figsize=(12, len(feature_names) * 0.3))  # Adjust width and height
        ax = sns.heatmap(layer_weights, annot=True, cmap='viridis',
                         xticklabels=['Neuron ' + str(j) for j in range(layer_weights.shape[1])],
                         yticklabels=feature_names if i == 0 else ['Neuron ' + str(j) for j in range(layer_weights.shape[0])],
                         cbar=True)
        plt.title(f'Layer {i+1} Weights')
        plt.xlabel('Neurons in Next Layer' if i < len(weights) - 1 else 'Output Neuron')
        plt.ylabel('Input Features' if i == 0 else 'Neurons in Previous Layer')

        plt.subplots_adjust(left=0.2, bottom=0.2)  # Adjust the bottom spacing to prevent cutting off labels
        plt.show()

def plot_model_coefficients(model, feature_names):
    # Extract coefficients from the model
    coefficients = model.coef_[0]
    
    # Combine the feature names and coefficients into a list of tuples and sort them by the coefficient value
    features_coefficients = sorted(zip(feature_names, coefficients), key=lambda x: x[1], reverse=True)
    
    sorted_features, sorted_coefficients = zip(*features_coefficients)
    
    # Plotting the coefficients
    plt.figure(figsize=(12, 10))  # Adjusted figure size
    plt.barh(range(len(sorted_features)), sorted_coefficients, color='skyblue')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Influence of Coefficients on Prediction')
    
    # Adjust layout to make room for feature names
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)  # Adjust this value as needed to prevent cutting off text
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plot confusion matrix.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - class_labels (array-like): List of class labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    tnr = tn / (tn + fp)  # True Negative Rate

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('Precision: ' + str(precision) + ' Recall: ' + str(recall) + ' F1: ' + str(f1))

    print(tpr, fpr, fnr, tnr)  # 1 1 1 1
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def generateROC(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    auc = roc_auc_score(y_test, y_pred)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
