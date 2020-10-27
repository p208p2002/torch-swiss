import unittest
import torch
import os,sys
sys.path.append('./')
import numpy as np
from torch_swiss.metrics import convert_classification_output_to_predicts
from torch_swiss.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1
# print(torch_swiss)

class TestMetrics(unittest.TestCase):
    def test_convert_classification_output_to_predicts(self):
        output = np.array([[1,2,3],[4,3,2],[3,5,4]])
        output = torch.from_numpy(output)
        label = np.array([2,0,1])
        predicts = convert_classification_output_to_predicts(output)
        predicts = predicts.numpy()
        self.assertEqual(str(label),str(predicts))

    def test_acc(self):
        #
        y_predict = torch.tensor([1,2,3])
        y_true = torch.tensor([1,2,3])
        self.assertEqual(compute_accuracy(y_predict,y_true),100.0)

        y_predict = torch.tensor([0,2,3,0])
        y_true = torch.tensor([1,2,3,4])
        self.assertEqual(compute_accuracy(y_predict,y_true),50.0)
        
        #
        y_predict = torch.tensor([0,0,0])
        y_true = torch.tensor([1,2,3])
        self.assertEqual(compute_accuracy(y_predict,y_true),0.0)
    
    def test_prec(self):
        # tp / (tp + fp)
        y_predict = torch.tensor([0,0,2,2,3,3,3,3])
        y_true    = torch.tensor([0,0,1,1,2,2,3,3])
        self.assertEqual(compute_precision(y_predict,y_true),37.5)

        y_predict = torch.tensor([3,3,3,3])
        y_true    = torch.tensor([0,0,1,1])
        self.assertEqual(compute_precision(y_predict,y_true),0.0)
    
    def test_recall(self):
        # tp / (tp + fn)
        y_predict = torch.tensor([0,0,2,2,3,3,3,3])
        y_true    = torch.tensor([0,0,1,1,2,2,3,3])
        self.assertEqual(compute_recall(y_predict,y_true),50.0)
    
    def test_f1(self):
        y_predict = torch.tensor([0,0,2,2,3,3,3,3])
        y_true    = torch.tensor([0,0,1,1,2,2,3,3])
        self.assertEqual(int(compute_f1(y_predict,y_true)),41)

if __name__ == '__main__':
    unittest.main()