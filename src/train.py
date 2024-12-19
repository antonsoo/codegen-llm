from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.data_prep import load_and_preprocess_data
import torch

def train_model(
        model_name="codellama/CodeLlama-7b-hf",
        dataset_name="bigcode/the-stack",
        data_dir="data/python",
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,  # Set to True if you want to push to the Hugging Face Hub
        **kwargs  # Additional keyword arguments for Trainer
):
    """Trains the code generation model.

    Args:
        model_name: The name of the pretrained model to use.
        dataset_name: Name of the dataset on Hugging Face Hub.
        data_dir: Specific directory within the dataset.
        output_dir: Where to save the trained model.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size for training.
        per_device_eval_batch_size: Batch size for evaluation.
        evaluation_strategy: When to evaluate during training ("steps" or "epoch").
        save_strategy: When to save the model ("steps" or "epoch").
        logging_dir: Where to store training logs.
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating.
        learning_rate: The initial learning rate.
        weight_decay: Weight decay for regularization.
        push_to_hub: Whether to push the trained model to the Hugging Face Hub.
        **kwargs: Additional keyword arguments for the Trainer.

    Returns:
        None
    """

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token if it's not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load and preprocess data
    train_dataset = load_and_preprocess_data(
        dataset_name=dataset_name, data_dir=data_dir, split="train", model_name=model_name
    )
    # Assuming you have a validation set
    eval_dataset = load_and_preprocess_data(
        dataset_name=dataset_name, data_dir=data_dir, split="validation", model_name=model_name
    )

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
        **kwargs  # Pass any additional keyword arguments to the Trainer
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
