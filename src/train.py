from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk, Dataset  # Import load_from_disk
from data_prep import create_subset_dataset
import torch

def train_model(
        model_name="salesforce/codegen-350m-multi", #"codellama/CodeLlama-7b-hf",
        dataset_name="bigcode/the-stack",
        data_dir="data/python",
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1, #4,
        per_device_eval_batch_size=1, #4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        gradient_accumulation_steps=16, #4,
        learning_rate=2e-5,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        #bf16=True,
        push_to_hub=False,
        **kwargs
):
    """Trains the code generation model."""

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create a subset of the dataset for training and evaluation
    subset_dir = create_subset_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        model_name=model_name,
        num_shards=5,
        subset_size=10000  # You can adjust the subset size
    )

    # Load the subset dataset from disk
    subset_dataset = load_from_disk(subset_dir)

    # Split the subset dataset into training and evaluation sets
    train_test_split = subset_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_dir=logging_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        push_to_hub=push_to_hub,
        **kwargs
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model()

