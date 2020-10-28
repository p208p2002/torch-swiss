from sklearn.metrics import precision_score as _precision_score, recall_score as _recall_score, f1_score as _f1_score, accuracy_score as _accuracy_score
from sklearn.metrics import matthews_corrcoef
import torch

def convert_classification_output_to_predicts(output):
    _, y_pred_indices = output.max(dim=1)
    return y_pred_indices.cpu().numpy()

def accuracy_score(*s_args,**p_args):
    return _accuracy_score(*s_args,**p_args)

def precision_score(*s_args,**p_args):
    return _precision_score(*s_args,average='macro',**p_args)

def recall_score(*s_args,**p_args):
    return _recall_score(*s_args,average='macro',**p_args)

def f1_score(*s_args,**p_args):
    return _f1_score(*s_args,average='macro',**p_args)
