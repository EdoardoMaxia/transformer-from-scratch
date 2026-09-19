import pytest
from tokenizer.tokenizer import CharTokenizer

def test_tokenizer_round_trip():
    # Setup
    text = "hello world!"
    chars = sorted(list(set(text)))
    tokenizer = CharTokenizer(chars)

    # Execution
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    # Assertion
    assert decoded == text, "Decoded text does not match the original input!"
    assert len(encoded) == len(text), "Token length does not match text length!"

def test_tokenizer_missing_char():
    # Setup with limited vocabulary
    tokenizer = CharTokenizer(['a', 'b', 'c'])

    # Assertion
    with pytest.raises(KeyError):
        tokenizer.encode("z")