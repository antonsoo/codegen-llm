from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import os

def create_subset_dataset(dataset_name="bigcode/the-stack", data_dir="data/python", split="train", model_name="codellama/CodeLlama-7b-hf", max_length=512, num_shards=5, subset_size=10000):
    """Loads a subset of The Stack dataset, tokenizes it, and saves it as a non-streaming dataset.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        data_dir: Specific directory within the dataset.
        split: Dataset split.
        model_name: The name of the pretrained model/tokenizer to use.
        max_length: The maximum sequence length for tokenization.
        num_shards: The number of shards to load from the dataset.
        subset_size: The desired size of the subset dataset.

    Returns:
        The path to the saved subset dataset.
    """

    # Load the dataset in a streaming fashion
    base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{data_dir}"
    data_files = [f"{base_url}/train-{i:05d}-of-00206.parquet" for i in range(num_shards)]

    # Create a streaming dataset
    dataset = load_dataset("parquet", data_files=data_files, split=split, streaming=True)
    print(f"Dataset loaded with {num_shards} shards in streaming mode.")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples["content"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_overflowing_tokens=False
        )
        return tokenized_output

    # Disable parallelism to avoid the PyGILState_Release error
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create a subset by taking the first few examples
    subset = dataset.take(subset_size)

    # Tokenize the subset
    tokenized_subset = subset.map(
        tokenize_function,
        batched=True,
        remove_columns=["content"]
    )

    # Convert to a regular dataset
    # First, convert the iterable to a list of dictionaries
    tokenized_subset_list = list(tokenized_subset)

    # Then, create the Dataset object from the list
    tokenized_subset = Dataset.from_dict({
        key: [item[key] for item in tokenized_subset_list] for key in tokenized_subset_list[0].keys()
    })

    # Save the subset dataset
    subset_dir = f"./subset_{num_shards}_shards_{subset_size}_samples"
    tokenized_subset.save_to_disk(subset_dir)
    print(f"Subset dataset saved to {subset_dir}")

    return subset_dir

if __name__ == "__main__":
    model_name = "codellama/CodeLlama-7b-hf"
    subset_dir = create_subset_dataset(model_name=model_name, num_shards=5, subset_size=10000)
                                                                                                         
