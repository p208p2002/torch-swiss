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

def count_word_piece_padding(tokenizer,word_tokens,item_start,item_end=None,init_padding_count=0):
    """
    tokenizer: transformer tokenizer
    word_tokens: word level tokens
    item_start: item start position
    item_end(optional): item end position
    init_padding_count(optional): when you have special token like [cls] as first, you can set to 1
    """
    entity_is_set = [False]*2
    padding = init_padding_count
    for i,token in enumerate(word_tokens):
        token_word_piece_length = len(tokenizer(token,add_special_tokens=False)['input_ids'])
         
        if(i == item_start and entity_is_set[0] == False):
            item_start+=padding
            entity_is_set[0] = True
            #
            if item_end is None:
                return item_start
        
        if(i == item_end and entity_is_set[1] == False):
            item_end+=padding
            entity_is_set[1] = True
        
        if(entity_is_set.count(True) == 2):
            return item_start,item_end

        padding += (token_word_piece_length - 1)
    assert False,'padding match error'

def balance_prob(all_gold_lablel_ids):
    unique_label_ids = list(set(all_gold_lablel_ids))    
    label_probs = []
    for label_id in range(len(unique_label_ids)):        
        label_id_count = all_gold_lablel_ids.count(label_id)
        label_probs.append(1./label_id_count)
    
    dataset_element_weights = [] # each element prob
    for label_id in all_gold_lablel_ids:                
        dataset_element_weights.append(label_probs[label_id])
    return dataset_element_weights