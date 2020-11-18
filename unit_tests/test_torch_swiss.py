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
        running_logger({"key1":0,"key2":"1"},title='test')
        rec_logger = LogRecorder()
        rec_logger.add_log(100.0,100.0)
        rec_logger.add_log(0.0,0.0)
        self.assertEqual(rec_logger.acc,50.0)
        self.assertEqual(rec_logger.loss,50.0)
        
    def test_device(self):
        from torch_swiss.device import detect_device,auto_apply_device
        model = TestModel()
        detect_device()
        auto_apply_device(model)
    
    def test_count_word_piece_padding(self):
        from torch_swiss import count_word_piece_padding
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
        test_label = {'relation_label': 'ethnic group', 'head_entity': {'name': 'Japan', 'pos': [140, 141], 'sent_id': 4, 'type': 'LOC', 'word_piece_pos': [168, 169]}, 'tail_entity': {'name': 'Japanese', 'pos': [167, 168], 'sent_id': 5, 'type': 'LOC', 'word_piece_pos': [201, 202]}, 'evidence_start_positions': [132, 162], 'word_piece_evidence_start_positions': [160, 196]}
        document_tokens = ['Lark', 'Force', 'was', 'an', 'Australian', 'Army', 'formation', 'established', 'in', 'March', '1941', 'during', 'World', 'War', 'II', 'for', 'service', 'in', 'New', 'Britain', 'and', 'New', 'Ireland', '.', 'Under', 'the', 'command', 'of', 'Lieutenant', 'Colonel', 'John', 'Scanlan', ',', 'it', 'was', 'raised', 'in', 'Australia', 'and', 'deployed', 'to', 'Rabaul', 'and', 'Kavieng', ',', 'aboard', 'SS', 'Katoomba', ',', 'MV', 'Neptuna', 'and', 'HMAT', 'Zealandia', ',', 'to', 'defend', 'their', 'strategically', 'important', 'harbours', 'and', 'airfields', '.', 'The', 'objective', 'of', 'the', 'force', ',', 'was', 'to', 'maintain', 'a', 'forward', 'air', 'observation', 'line', 'as', 'long', 'as', 'possible', 'and', 'to', 'make', 'the', 'enemy', 'fight', 'for', 'this', 'line', 'rather', 'than', 'abandon', 'it', 'at', 'the', 'first', 'threat', 'as', 'the', 'force', 'was', 'considered', 'too', 'small', 'to', 'withstand', 'any', 'invasion', '.', 'Most', 'of', 'Lark', 'Force', 'was', 'captured', 'by', 'the', 'Imperial', 'Japanese', 'Army', 'after', 'Rabaul', 'and', 'Kavieng', 'were', 'captured', 'in', 'January', '1942', '.', 'The', 'officers', 'of', 'Lark', 'Force', 'were', 'transported', 'to', 'Japan', ',', 'however', 'the', 'NCOs', 'and', 'men', 'were', 'unfortunately', 'torpedoed', 'by', 'the', 'USS', 'Sturgeon', 'while', 'being', 'transported', 'aboard', 'the', 'Montevideo', 'Maru', '.', 'Only', 'a', 'handful', 'of', 'the', 'Japanese', 'crew', 'were', 'rescued', ',', 'with', 'none', 'of', 'the', 'between', '1,050', 'and', '1,053', 'prisoners', 'aboard', 'surviving', 'as', 'they', 'were', 'still', 'locked', 'below', 'deck', '.']        
        new_start,new_end = count_word_piece_padding(tokenizer,document_tokens,140,141,init_padding_count=1)
        
        self.assertEqual(new_start,168)
        self.assertEqual(new_end,169)
        
        new_start,new_end = count_word_piece_padding(tokenizer,document_tokens,167,168,init_padding_count=1)
        self.assertEqual(new_start,201)
        self.assertEqual(new_end,202)


if __name__ == '__main__':
    unittest.main()
