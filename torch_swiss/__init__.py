import torch
import torch.nn as nn

def convert_classification_output_to_predicts(output):
    _, y_pred_indices = output.max(dim=1)
    return y_pred_indices.cpu().numpy()

def split_dataset(full_dataset,split_rate = 0.8):
    train_size = int(split_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset
