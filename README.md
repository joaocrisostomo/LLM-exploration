Fine-Tuning Llama Vision 3.2 for Document Processing
This repository contains a Jupyter notebook demonstrating how to fine-tune the Llama Vision 3.2 model for invoice processing using the Unsloth framework. The notebook guides you through the process of setting up the environment, preparing the data, fine-tuning the model, and using it for extracting payment information from invoices.

Overview
Llama Vision 3.2 is a multimodal vision model optimized for image reasoning and visual recognition tasks. By fine-tuning it with Unsloth, we tailor it for invoice processing, improving its ability to handle diverse invoice layouts and extract critical details such as license plate numbers, event dates, and amounts.

This guide walks you through every step, from preparing the dataset to fine-tuning the model and testing its capabilities on real-world invoice data.

Tools and Libraries Used
Llama Vision 3.2: A state-of-the-art vision model designed for image reasoning and multimodal tasks.
Unsloth: A framework that simplifies the fine-tuning process for large models like Llama Vision, optimizing memory usage and training speed.
PyTorch: The deep learning framework used to train the model.
Pandas: Used for data manipulation and preparation.
PIL (Pillow): For image processing.
Transformers Library: Used for tokenization and model interaction.
Requirements
To run this project, you need the following dependencies:

Python 3.x
Jupyter Notebook
torch
transformers
unsloth
pandas
PIL (Pillow)
s3fs (for S3 file downloads)
Google Colab (optional for running on free GPUs)
You can install the necessary dependencies using pip:

bash
Copy
Edit
pip install torch transformers unsloth pandas pillow s3fs
Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/llama-vision-invoice-processing.git
cd llama-vision-invoice-processing
Download the dataset:
Prepare a dataset of invoices (PDFs) and manually processed data in a pandas DataFrame. The DataFrame should include columns such as license_plate, event_date, and amount.

Run the notebook:
Open the Jupyter notebook and follow the instructions in each cell to set up the environment, preprocess data, and fine-tune the model.

bash
Copy
Edit
jupyter notebook fine_tune_llama_vision_3_2.ipynb
Steps in the Notebook
Data Preparation:
We convert PDF invoices into images and prepare the corresponding text captions using pandas DataFrames.

Model Fine-Tuning:
The model is fine-tuned using the Unsloth framework to optimize performance for the specific task of extracting data from invoices.

Model Inference:
After fine-tuning, we use the model to perform inference on new invoices and extract the desired information, such as license plate numbers, event dates, and amounts.

Model Saving:
Once fine-tuning is complete, the trained model is saved locally for later use.

Usage
Once you've fine-tuned the model, you can use it for inference on new invoices by running the following code:

python
Copy
Edit
from unsloth import FastVisionModel
from transformers import TextStreamer

# Load the fine-tuned model
model = FastVisionModel.from_pretrained("path_to_your_model")
tokenizer = model.get_tokenizer()

# Prepare the image and text input for inference
image = "path_to_invoice_image"
instruction = "You are an expert of extracting fines and tolls information from invoices."

messages = [
    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}
]

inputs = tokenizer(
    image, 
    messages, 
    add_special_tokens=False, 
    return_tensors="pt"
).to("cuda")

# Generate and print the output
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=200)
print(output)
Contributing
If you have suggestions, improvements, or new features you'd like to contribute, feel free to submit a pull request or open an issue. Contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Llama Vision 3.2
Unsloth Framework
Pandas
PyTorch
Hugging Face Transformers
