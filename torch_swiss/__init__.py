import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import os

def split_dataset(full_dataset,split_rate = 0.8):
    train_size = int(split_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def running_logger(log_dict,title=None,end='\r'):
    """
    running_logger({"batch":i+1, "loss":"%3.5f"%test_log_recorder.loss, "acc":"%3.5f"%test_log_recorder.acc},'test',end='\r')
    """
    if title is not None:
        print(title,end=' ')
    for key in log_dict.keys():
        print("%s:"%key,log_dict[key],end=' ')
    print(end=end)

def make_confusion_matrix(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)

class LogRecorder():
    def __init__(self):
        self.i = 0
        self.running_acc = 0.0
        self.running_loss = 0.0
        self._y_trues = []
        self._y_predicts = []

    def reset(self):
        self.__init__()

    def add_log(self,batch_acc,batch_loss):
        self.running_acc += (batch_acc - self.running_acc)/(self.i+1)
        self.running_loss += (batch_loss - self.running_loss)/(self.i+1)
        self.i += 1
    
    def save_predicts_and_trues(self,logits,label):
        label = label.to('cpu')
        y_predicts = torch.argmax(logits,dim=1).to('cpu')
        assert y_predicts.size(0) == label.size(0)

        for y_p,y_t in zip(y_predicts,label):
            self._y_predicts.append(y_p.item())
            self._y_trues.append(y_t.item())
    
    @property
    def acc(self):
        return self.running_acc
    
    @property
    def loss(self):
        return self.running_loss

class ModelHolder():
    def __init__(self,model):
        self.model = model
        
    def __enter__(self):
        return self.model
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            pass

def detect_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def auto_aply_device(model):
    device = detect_device()
    if torch.cuda.device_count() >1:
        model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    model.to(device)
    print("using device",device)
    return model

