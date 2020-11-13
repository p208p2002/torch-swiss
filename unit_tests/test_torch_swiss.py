import unittest
import torch
import os,sys
sys.path.insert(0,'./') # use local torch_swiss
import numpy as np
import torch_swiss
from unit_tests.torch_model import TestModel
print(torch_swiss)

class TestTorchSwiss(unittest.TestCase):
    def test_convert_classification_output_to_predicts(self):
        from torch_swiss import convert_classification_output_to_predicts

        output = np.array([[1,2,3],[4,3,2],[3,5,4]])
        output = torch.from_numpy(output)
        label = np.array([2,0,1])
        predicts = convert_classification_output_to_predicts(output)
        self.assertEqual(str(label),str(predicts))
    
    def test_model_holder(self):
        from torch_swiss.model_holder import ModelHolder

        os.system('rm -rf .model_holder')
        model = TestModel()
        with ModelHolder(model) as (holder,model):
            holder.save_checkpoint('manual_checkpoint.bin')

        self.assertTrue(os.path.isdir('.model_holder'))
        self.assertTrue(os.path.isfile('.model_holder/manual_checkpoint.bin'))
    
    def test_logger(self):
        from torch_swiss.logger import LogRecorder,running_logger
        LogRecorder()
        running_logger({"key1":0,"key2":"1"},title='test')
    
    def test_device(self):
        from torch_swiss.device import detect_device,auto_apply_device
        model = TestModel()
        detect_device()
        auto_apply_device(model)
        


if __name__ == '__main__':
    unittest.main()
