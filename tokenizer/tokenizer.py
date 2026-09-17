from typing import List, Dict

class CharTokenizer:
    """
    Character tier Tokenizer for the text elaboration.
    """

    def __init__(self, chars: List[str]) -> None:
        self.chars = chars
        self.vocab_size = len(chars)

        # Lookup tables
        self.stoi: Dict[str, int] = {ch:i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i:ch for i, ch in enumerate(self.chars)}

    #=========================================
    # ENCODER
    #=========================================
    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of int.
        """
        # If a character is not present inside the vocabolary, it will raise a KeyError.
        return [self.stoi[c] for c in text]


    #=========================================
    # DENCODER
    #=========================================
    def decode(self, tokens: List[int]) -> str:
        """
        Convert a list of int to a string.
        """
        return ''.join([self.itos[i] for i in tokens])

    #=========================================
    # TEST 
    #=========================================
    def test_token(self, test_str: str = "Hello World!") -> None:
        try:
            encoded = self.encode(test_str)
            decoded = self.decode(encoded)
            print(f"\nTest Encoding ('{test_str}') : {encoded}")
            print(f"\nTest Decoding : {decoded}")
        except KeyError as e:
            print(f"\nError: The character {e} is not present inside the training vocabulary!")