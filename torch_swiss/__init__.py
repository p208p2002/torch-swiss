import torch
import torch.nn as nn
import os,sys,time

def split_dataset(dataset,split_rate = 0.8):
    """
    split dataset into two

    Args:
        dataset (class): pytorch dataset 
        split_rate (float): first dataset size
    
    Returns:
        tuple(class): train_dataset, test_dataset
    """
    train_size = int(split_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def compute_word_piece_padding(tokenizer,word_tokens,item_start,item_end=None,init_padding_count=0):
    """
    compute new word padding after doing word-piece

    Args:
        tokenizer (class): transformer tokenizer
        word_tokens (list[str]): word level tokens
        item_start (int): item start position
        item_end (int, optional): item end position
        init_padding_count (int, optional): when you have special token like [cls] as first, you can set to 1

    Returns:
        int or tuple(int): 
            `item_start`, if only `item_start` given.\n
            `item_start` and `item_start`, if both `item_start` and `item_start` given
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
            item_end+=(padding+token_word_piece_length-1)
            entity_is_set[1] = True
        
        if(entity_is_set.count(True) == 2):
            return item_start,item_end

        padding += (token_word_piece_length - 1)
    assert False,'padding match error'

def balance_prob(all_gold_lablel_ids):
    """
    set balance probability to each label for process imlalance

    Args:
        all_gold_lablel_ids (list[int]): all gold_lablel ids
    
    Returns:
        list[float]: each label probability
    """
    unique_label_ids = list(set(all_gold_lablel_ids))    
    label_probs = []
    for label_id in range(len(unique_label_ids)):        
        label_id_count = all_gold_lablel_ids.count(label_id)
        label_probs.append(1./label_id_count)
    
    dataset_element_weights = [] # each element prob
    for label_id in all_gold_lablel_ids:                
        dataset_element_weights.append(label_probs[label_id])
    return dataset_element_weights

def save_sys_argv(save_path='./',save_name='args.log'):
    """
    save sys argv "if has"

    Args:
        save_path (str): path to save
        save_name (str): log file's name
    
    Returns:
        None
    """
    os.makedirs(os.path.join(save_path),exist_ok=True)
    with open(os.path.join(save_path,save_name),'a') as f:
        f.write(time.ctime()+'\t '+' '.join(sys.argv)+'\n')
        
def set_model_requires_grad(model,should_requires_grad):
    """
    set model.parameters should requires gradient or not

    Args:
        model (class<torch.nn.Module>): pytorch model
        should_requires_grad (bool): `True` or `False`
    
    Returns:
        None
    """
    model = model.module if hasattr(model, "module") else model
    for param in model.parameters():
        param.requires_grad = should_requires_grad
