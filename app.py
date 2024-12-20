import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your model path or Hugging Face model ID
model_path = "./results"  # Use this if you saved the model locally
# model_path = "your_huggingface_username/your_model_name"  # Use this if you pushed your model to the Hugging Face Hub

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad token if it's not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate code function
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    # Ensure that the input tensor's device matches the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_k=50, top_p=0.95)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# Create Gradio interface
iface = gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(lines=5, label="Enter your prompt:"),
    outputs=gr.Textbox(label="Generated code:"),
    title="Code Generation Demo",
    description="Enter a prompt to generate code with the fine-tuned model."
)

# Launch the demo
iface.launch(share=True) # Set share=True to create a temporary public URL (useful for testing and sharing)
