import torch
from tokenizer.tokenizer import CharTokenizer

class TrainData:
    """
    Prepare and Manage the tensors for the training and validation.
    """

    # Dependency Injection
    def __init__(self, text: str, tokenizer: CharTokenizer) -> None:
        self.text = text
        self.tokenizer = tokenizer

        self.data: torch.Tensor = torch.Tensor([])
        self.train_data: torch.Tensor = torch.Tensor([])
        self.val_data: torch.Tensor = torch.Tensor([])

    def text_to_tensor(self) -> None:
        """
        This method converts a text to a Pythorch tensor using the tokenizer Class
        """

        encoded_text = self.tokenizer.encode(self.text)
        self.data = torch.tensor(encoded_text, dtype=torch.long)

    def split_train_val_data(self) -> None:
        """
        Divide the data: 90% for training; 10% for validation
        """
        # Safety check: if we don't already converted the data, we do it here
        if self.data.numel() == 0:
            self.text_to_tensor()

        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        print(f"Training set dimension: {len(self.train_data)} token")
        print(f"Validation set dimension: {len(self.val_data)} token")