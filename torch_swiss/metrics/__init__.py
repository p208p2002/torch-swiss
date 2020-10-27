from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def convert_classification_output_to_predicts(output):
    _, y_pred_indices = output.max(dim=1)
    return y_pred_indices

def compute_accuracy(y_pred, y_true):
    n_correct = torch.eq(y_pred, y_true).sum().item()
    return n_correct / len(y_pred) * 100

def compute_precision(y_pred, y_true):
    return precision_score(y_true, y_pred,average='macro') * 100

def compute_recall(y_pred, y_true):
    return recall_score(y_true, y_pred,average='macro') * 100

def compute_f1(y_pred, y_true):
    return f1_score(y_true, y_pred,average='macro') * 100
