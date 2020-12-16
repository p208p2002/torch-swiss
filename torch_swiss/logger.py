import torch

def running_logger(log_dict,title=None,end='\r'):
    """
    running_logger({"batch":i+1, "loss":"%3.5f"%test_log_recorder.loss, "acc":"%3.5f"%test_log_recorder.acc},'test',end='\r')
    """
    if title is not None:
        print(title,end=' ')
    for key in log_dict.keys():
        print("%s:"%key,log_dict[key],end=' ')
    print(end=end)

class LogRecorder():
    def __init__(self):
        self.i = 0
        self.running_acc = 0.0
        self.running_loss = 0.0
        self._y_trues = []
        self._y_predicts = []

    def reset(self):
        self.__init__()

    def add_log(self,batch_acc=0.0,batch_loss=0.0):
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
