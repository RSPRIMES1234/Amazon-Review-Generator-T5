# Amazon-Review-Generator-t5
This project uses a fine-tuned T5 model [RSPRIMES1234/Amazon-Review-Generator-T5](https://huggingface.co/RSPRIMES1234/Amazon-Review-Generator-T5) to generate Amazon product reviews based on the product title and star rating.  
## Description  

This Python script fine-tunes a T5 model on Amazon review data for software products. It can then generate realistic product reviews given a product title and star rating.

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- Pandas
- NumPy
- CUDA-capable GPU (optional, but recommended for faster processing)

## Installation

1. Clone this repository:
```python
git clone https://github.com/RSPRIMES1234/Amazon-Review-Generator-T5
```
2. Install the required packages:
```python
pip install torch transformers datasets pandas numpy
```

## Usage

1. Prepare the dataset:
- The script uses the "McAuley-Lab/Amazon-Reviews-2023" dataset from Hugging Face.
- It filters for verified purchases and reviews longer than 100 characters.

2. Fine-tune the model:
- Uncomment the `trainer.train()` and `trainer.save_model(TRAINING_OUTPUT)` lines in the script to train the model.
- This step may take several hours depending on your hardware.

3. Generate reviews:
- The script will automatically generate three sample reviews after loading the fine-tuned model.
- You can modify the `random_products` and generate more reviews as needed.

## Code Structure

- Data loading and preprocessing
- Model and tokenizer initialization
- Training setup (currently commented out)
- Review generation function
- Sample review generation

## GPU Acceleration

The script is set up to use GPU acceleration if available. It will automatically detect if a CUDA-capable GPU is present and use it for both training (if uncommented) and inference.

## Customization

- You can change the `dataset_category` variable to fine-tune on different product categories.
- Adjust the `TrainingArguments` to modify the training process.
- The `generate_review` function can be customized to change the generation parameters.
