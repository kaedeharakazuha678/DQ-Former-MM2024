import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

def print_args(args):
    """Print the arguments of the model"""
    header = "Arguments"
    separator_length = 30 + len(header) + 30
    separator = "=" * separator_length
    
    print(separator)
    print(f"|{' ' * 29}{header}{' ' * 29}|")
    for arg in vars(args):
        arg_str = f"|    {arg}: {getattr(args, arg)}"
        padding = ' ' * (separator_length - len(arg_str))
        print(f"{arg_str}{padding}|")
    print(separator)

def get_new_filename(directory, filename):
    # If the file exists, find a new filename by appending a suffix
    base, ext = os.path.splitext(filename)
    index = 1
    month_date = time.strftime("%m%d", time.localtime(time.time()))
    new_filename = f"{base}_{month_date}_{index}{ext}"

    # Keep incrementing the index until a non-existing filename is found
    while os.path.exists(os.path.join(directory, new_filename)):
        index += 1
        new_filename = f"{base}_{month_date}_{index}{ext}"

    return new_filename

def print_trainable_params(model):
    """Print the trainable parameters of the model"""
    print("="*30+"Model Structure"+"="*30)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'|   Total Parameters: {total_params}   |   Trainable Parameters: {trainable_params}  |   Trainable rate: {trainable_params/total_params}   |')
    print("="*30+"Trainable Parameters"+"="*30)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("="*70)

def show_gate_weights(model):
    """Show the gate weights of the model"""
    gate_data = []
    for name, param in model.named_parameters():
        if 'ff_gate' in name:
            layer_name = name.split('.')[2] 
            gate_value = param.data.item() 
            gate_data.append({'Layer': layer_name, 'Gate_name':'ff_gate','Gate Value': gate_value})
        if 'attn_gate' in name:
            layer_name = name.split('.')[2]
            gate_value = param.data.item()
            gate_data.append({'Layer': layer_name, 'Gate_name':'attn_gate','Gate Value': gate_value})
    df = pd.DataFrame(gate_data)
    return df


#region single confusion matrix plot
# Description:
#  This function is used to plot the confusion matrix.
# Input:
#  y_pred: the predicted labels
#  y_label: the true labels
#  labels: the labels
#  target_names: the target names
#  title: the title of the confusion matrix
#  save_path: the path to save the confusion matrix
#  normalize: the normalize method
# Output:
#  cm: the confusion matrix
# Example:
#  cm = confusion_matrix_plot(y_pred, y_label, labels, target_names, title, save_path, normalize=None)

def confusion_matrix_plot(y_pred, y_label, labels, target_names, title, save_path, normalize=None):
    plt.figure(dpi=100)
    cm = confusion_matrix(y_label, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='YlGnBu')
    disp.ax_.set_title(title)
    disp.ax_.set_xlabel('Predicted Label')
    disp.ax_.set_ylabel('True Label')
    plt.show()
    plt.savefig(os.path.join(save_path, title+'.png'), dpi=10000)
    return cm

#endregion


#region draw the confusion matrix map
# Description:
#  This function is used to draw the confusion matrix map.
# Input:
#  all_preds: the predicted labels
#  all_labels: the true labels
#  all_preds_l: the predicted labels of linguistic
#  all_preds_a: the predicted labels of audio
# Output:
#  array: the confusion matrix map
# Example:
#  array = evaluate_array(all_preds, all_labels, all_preds_l, all_preds_a, pred_condition=True)
#  matrix = array_to_matrix(array)
#  plot_heatmap(matrix, ax, title)
#  plot_confusion(all_preds, all_preds_l, all_preds_a, all_labels, save_path)


def evaluate_array(all_preds, all_labels, all_preds_l, all_preds_a, pred_condition=True):
    """Evaluates the predictions and returns an 8x1 array."""
    def scenario_index(pred, pred_l, pred_a, label):
        """Returns the index of the array."""
        return (int(pred_l == label) << 1) | int(pred_a == label)

    # Initialize an 8x1 array with zeros
    array = np.zeros(4, dtype=int)

    # Loop over each value in the arrays
    for pred, label, pred_l, pred_a in zip(all_preds, all_labels, all_preds_l, all_preds_a):
        if (pred == label) == pred_condition:
            i = scenario_index(pred, pred_l, pred_a, label)
            array[i] += 1

    return array

def array_to_matrix(arr):
    """Converts the 1x4 array to a 2x2 matrix."""
    return arr.reshape(2, 2)

def plot_heatmap(matrix, ax, title):
    plt.figure(dpi=10000)
    """Utility function to plot a heatmap."""
    cax = ax.imshow(matrix, cmap='YlGnBu', interpolation='nearest', vmin=0, vmax=matrix.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha='center', va='center', color='black' if matrix[i, j] < matrix.max() / 2 else 'white', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['✘', '✔'], fontsize=16)
    ax.set_yticklabels(['✘', '✔'], fontsize=16, rotation=90)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Audio Predict Result', fontsize=12)
    ax.set_ylabel('Linguistic Predict Result', fontsize=12)
    cbar = ax.figure.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    return cax


def plot_confusion(all_preds, all_preds_l, all_preds_a, all_labels, save_path):
    true_pred_array = evaluate_array(all_preds, all_labels, all_preds_l, all_preds_a, pred_condition=True)
    false_pred_array = evaluate_array(all_preds, all_labels, all_preds_l, all_preds_a, pred_condition=False)

    true_pred_matrix = array_to_matrix(true_pred_array)
    false_pred_matrix = array_to_matrix(false_pred_array)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    plot_heatmap(true_pred_matrix, ax[0], 'Correct Condition')
    plot_heatmap(false_pred_matrix, ax[1], 'Error Condition')

    fig.subplots_adjust(right=0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=10000)
    plt.show()
#endregion