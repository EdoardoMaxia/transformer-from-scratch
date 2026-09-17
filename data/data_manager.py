import os
import urllib.request
from typing import List


class TextDataManager:
    """
    Dataset Manager for Language Models training.
    """

    TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(self, url: str = TINY_SHAKESPEARE_URL, filename: str = "tinysheakespeare.txt") -> None:
        self.url = url
        self.filename = filename

        self.text: str = ""
        self.chars: List[str] = []
        self.vocab_size: int = 0

    #==============================
    # 1. Load the dataset
    #==============================
    def load_dataset(self) -> None:
        """
        Download the dataset if doesn't exist in local, then uploads it in memory
        """
        if not os.path.exists(self.filename):
            print(f"Downloading the dataset:\n{self.url}\n....")
            urllib.request.urlretrieve(self.url, self.filename)
        else:
            print(f"The dataset {self.filename} already exists in the repo.")

        with open(self.filename, 'r', encoding='utf-8') as f:
            self.text = f.read()

        print(f"The total lenght of the dataset: {len(self.text)} character.")

    
    #==============================
    # 2. Load the dataset
    #==============================
    def create_vocabulary(self) -> None:
        """
        Find all unique characters inside the text.
        """
        # Safety check: if the text is empty, load it
        if not self.text:
            self.load_dataset()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        print(f"Vocabolary with ({self.vocab_size}) characters: {''.join(self.chars)}")

    