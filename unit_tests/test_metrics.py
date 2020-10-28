import unittest
import torch
import os,sys
sys.path.insert(0,'./') # use local torch_swiss
import numpy as np
import torch_swiss
from torch_swiss.metrics import convert_classification_output_to_predicts
from torch_swiss.metrics import accuracy_score,precision_score,recall_score,f1_score
print(torch_swiss)

class TestMetrics(unittest.TestCase):
    def test_convert_classification_output_to_predicts(self):
        output = np.array([[1,2,3],[4,3,2],[3,5,4]])
        output = torch.from_numpy(output)
        label = np.array([2,0,1])
        predicts = convert_classification_output_to_predicts(output)
        self.assertEqual(str(label),str(predicts))

    def test_acc(self):
        #
        y_predict = torch.tensor([1,2,3])
        y_true = torch.tensor([1,2,3])
        self.assertEqual(accuracy_score(y_pred=y_predict, y_true= y_true), 1)

        y_predict = torch.tensor([0,2,3,0])
        y_true = torch.tensor([1,2,3,4])
        self.assertEqual(accuracy_score(y_pred=y_predict, y_true= y_true), 0.5)
        
        #
        y_predict = torch.tensor([0,0,0])
        y_true = torch.tensor([1,2,3])
        self.assertEqual(accuracy_score(y_pred=y_predict, y_true= y_true), 0.0)
    
    def test_prec(self):
        # tp / (tp + fp)
        y_predict = torch.tensor([0,0,2,2,3,3,3,3])
        y_true    = torch.tensor([0,0,1,1,2,2,3,3])
        self.assertEqual(precision_score(y_pred=y_predict, y_true= y_true), 0.375)

        y_predict = torch.tensor([3,3,3,3])
        y_true    = torch.tensor([0,0,1,1])
        self.assertEqual(precision_score(y_pred=y_predict, y_true= y_true), 0.0)
    
    def test_recall(self):
        # tp / (tp + fn)
        y_predict = torch.tensor([0,0,2,2,3,3,3,3])
        y_true    = torch.tensor([0,0,1,1,2,2,3,3])
        self.assertEqual(recall_score(y_pred=y_predict, y_true= y_true) ,0.5)
    
    def test_f1(self):
        y_predict = torch.tensor([0,0,2,2,3,3,3,3])
        y_true    = torch.tensor([0,0,1,1,2,2,3,3])
        self.assertEqual(int(f1_score(y_pred=y_predict, y_true= y_true)*100), 41)

if __name__ == '__main__':
    unittest.main()