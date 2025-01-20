# Fine-Tuning Llama Vision 3.2 for Document Processing

![Llama Vision 3.2](![Uploading Screenshot 2025-01-20 at 23.16.20.png…]()
)

This repository contains a Jupyter notebook demonstrating how to fine-tune the Llama Vision 3.2 model for invoice processing using the Unsloth framework. The notebook guides you through the process of setting up the environment, preparing the data, fine-tuning the model, and using it for extracting payment information from invoices.

## Overview

Llama Vision 3.2 is a multimodal vision model optimized for image reasoning and visual recognition tasks. By fine-tuning it with Unsloth, we tailor it for invoice processing, improving its ability to handle diverse invoice layouts and extract critical details such as license plate numbers, event dates, and amounts.

This guide walks you through every step, from preparing the dataset to fine-tuning the model and testing its capabilities on real-world invoice data.

## Tools and Libraries Used

- **Llama Vision 3.2**: A state-of-the-art vision model designed for image reasoning and multimodal tasks.
- **Unsloth**: A framework that simplifies the fine-tuning process for large models like Llama Vision, optimizing memory usage and training speed.
- **PyTorch**: The deep learning framework used to train the model.
- **Pandas**: Used for data manipulation and preparation.
- **PIL (Pillow)**: For image processing.
- **Transformers Library**: Used for tokenization and model interaction.

## Requirements

To run this project, you need the following dependencies:

### Setup

1. **Install dependencies**:

    ```bash
    pip install torch transformers unsloth pandas pillow s3fs
    ```

2. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/llama-vision-invoice-processing.git
    cd llama-vision-invoice-processing
    ```

3. **Download the dataset**:  
   Prepare a dataset of invoices (PDFs) and manually processed data in a pandas DataFrame. The DataFrame should include columns such as `license_plate`, `event_date`, and `amount`.

4. **Run the notebook**:  
   Open the Jupyter notebook and follow the instructions in each cell to set up the environment, preprocess data, and fine-tune the model.

## Model Fine-Tuning

The model is fine-tuned using the Unsloth framework to optimize performance for the specific task of extracting data from invoices. The notebook guides you through setting up the training process and adjusting hyperparameters to improve the model's accuracy.

## Model Inference

After fine-tuning, we use the model to perform inference on new invoices and extract the desired information, such as license plate numbers, event dates, and amounts. The model can process new data and generate structured output.

## Model Saving

Once fine-tuning is complete, the trained model is saved locally for later use. This allows you to reload and use the model for inference on new invoices without the need for re-training.

## Usage

Once you've fine-tuned the model, you can use it for inference on new invoices by running the following code:

```python
# Example of inference
image = "path_to_invoice_image"
model_output = model.predict(image)
print(model_output)
