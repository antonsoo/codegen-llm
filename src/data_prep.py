from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def load_and_preprocess_data(dataset_name="bigcode/the-stack", data_dir="data/python", split="train", model_name="codellama/CodeLlama-7b-hf", max_length=512):
    """Loads, tokenizes, and preprocesses The Stack dataset for code generation.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        data_dir: Specific directory within the dataset (e.g., for a specific language).
        split: Dataset split (e.g., "train", "validation", "test").
        model_name: The name of the pretrained model/tokenizer to use.
        max_length: The maximum sequence length for tokenization.

    Returns:
        The preprocessed dataset ready for training.
    """

    # Load the dataset
    dataset = load_dataset(dataset_name, data_dir=data_dir, split=split)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token if it's not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize with padding and truncation
        tokenized_output = tokenizer(
            examples["content"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_overflowing_tokens=False  # You can set this to True and handle it appropriately if needed
        )
        return tokenized_output

    # Map the tokenization function to the dataset
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Adjust the number of processes as needed
        remove_columns=["content"],  # Remove the original text column
        load_from_cache_file=True,  # Consider setting to False if you are debugging or modifying the tokenization process
    )

    return tokenized_datasets

# Example usage (you can run this in your Colab notebook)
if __name__ == "__main__":
    model_name = "codellama/CodeLlama-7b-hf"
    tokenized_data = load_and_preprocess_data(model_name=model_name)
    print(f"Dataset tokenized and preprocessed.")
    print(tokenized_data[0]) # Print first example
