from data.data_manager import TextDataManager
from tokenizer.tokenizer import CharTokenizer
from data.train_data import TrainData

def main():
    print("--- 1. LOADING DATA ---")
    data_manager = TextDataManager()
    data_manager.load_dataset()
    data_manager.create_vocabulary()

    print("\n--- 2. TOKENIZING ---")
    tokenizer = CharTokenizer(data_manager.chars)
    tokenizer.test_token("Hello LLM!")

    print("\n--- 3. BUILDING TENSOR ---")
    tensor_manager = TrainData(text=data_manager.text, tokenizer=tokenizer)
    tensor_manager.text_to_tensor()
    tensor_manager.split_train_val_data()

    print("\nSetup completed!")


if __name__ == "__main__":
    main()