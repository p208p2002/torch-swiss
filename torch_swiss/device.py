import torch
import torch.nn as nn

def detect_device():
    """
    detect cuda available or not

    Returns:
        class <torch.device>: torch device
    """
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def auto_apply_device(model):
    """
    auto apply model into muilt-gpus

    Args:
        model (class<torch.nn.Module>): pytorch model
    
    Returns:
        model (class<torch.nn.Module>): pytorch model(into nn.DataParallel)
    """
    device = detect_device()
    if torch.cuda.device_count() >1:
        model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    model.to(device)
    print("using device",device)
    return model