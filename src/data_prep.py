from datasets import load_dataset

def load_and_preprocess_data(dataset_name="bigcode/the-stack", data_dir="data/python", split="train"):
    """Loads and preprocesses the dataset.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        data_dir: Specific directory within the dataset (e.g., for a specific language).
        split: Dataset split (e.g., "train", "validation", "test").

    Returns:
        The preprocessed dataset.
    """

    dataset = load_dataset(dataset_name, data_dir=data_dir, split=split)

    # Add preprocessing steps here (tokenization, etc.)
    # ... (You'll add more detailed code in the next steps)

    return dataset

if __name__ == "__main__":
    data = load_and_preprocess_data()
    print(f"Dataset loaded with {len(data)} samples.")
