
import os
import torch
import datetime

class ModelHolder():
    def __init__(self,model):
        self.model = model
    
    def save_checkpoint(self,save_name = None):
        if not os.path.isdir('.model_holder'):
            os.mkdir('.model_holder')

        if save_name is None:
            save_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.bin")
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save, '.model_holder/%s'%save_name)
    
    def __enter__(self):
        return self,self.model
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            print("Catch KeyboardInterrupt, model will auto save in .model_holder")
        
        self.save_checkpoint()