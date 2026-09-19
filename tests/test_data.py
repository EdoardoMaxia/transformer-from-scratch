import torch
from tokenizer.tokenizer import CharTokenizer
from data.train_data import TrainData

def test_train_val_split_dimensions():
    # Setup
    text = "abcdefghij"
    chars = sorted(list(set(text)))
    tokenizer = CharTokenizer(chars)

    tensor_manager = TrainData(text=text, tokenizer=tokenizer)
    tensor_manager.text_to_tensor()
    tensor_manager.split_train_val_data()

    # Assertion data type
    assert isinstance(tensor_manager.data, torch.Tensor)
    assert tensor_manager.data.dtype == torch.long

    # Assertion split
    assert len(tensor_manager.train_data) == 9
    assert len(tensor_manager.val_data) == 1
    assert len(tensor_manager.train_data) + len(tensor_manager.val_data) == len(tensor_manager.data)