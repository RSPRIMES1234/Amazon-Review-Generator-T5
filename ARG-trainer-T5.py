#This Code is for the people who want to train their own model 
#This Code is not tested because of the time it would have taken to rebuild the model again , so it might have some error 
import torch
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# Specify the category of Amazon products to use
dataset_category = "Software"

# Load raw metadata and review datasets from McAuley-Lab's Amazon-Reviews-2023 dataset
meta_ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{dataset_category}", split='full', trust_remote_code=True).to_pandas()[['parent_asin', 'title']]
review_ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{dataset_category}", split='full', trust_remote_code=True).to_pandas()[['parent_asin', 'rating', 'text', 'verified_purchase']]

# Merge metadata and review datasets, filter for verified purchases and reviews longer than 100 characters, and sample 100,000 rows
ds = meta_ds.merge(review_ds, on='parent_asin', how='inner').drop(columns="parent_asin")
ds = ds.rename(columns={"rating":"star_rating", "title":"product_title", "text":"review_body"})
ds = ds[ds['verified_purchase'] & (ds['review_body'].map(len) > 100)].sample(100_000)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(ds)

# Encode star_rating column
dataset = dataset.class_encode_column("star_rating")

# Split dataset into training and testing sets, stratified by star_rating
dataset = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="star_rating")
train_dataset = dataset['train']
test_dataset = dataset['test']

# Initialize tokenizer using 't5-base' model
MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Define function to preprocess data for training and evaluation
def preprocess_data(examples):
    # Create prompts and responses for the model
    examples['prompt'] = [f"review: {product_title}, {star_rating} Stars!" for product_title, star_rating in zip(examples['product_title'], examples['star_rating'])]
    examples['response'] = [f"{review_body}" for review_body in examples['review_body']]

    # Tokenize inputs and targets, ensuring padding and truncation
    inputs = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=128)
    targets = tokenizer(examples['response'], padding='max_length', truncation=True, max_length=128)

    # Set -100 at padding positions of target tokens (to ignore loss calculation)
    target_input_ids = []
    for ids in targets['input_ids']:
        target_input_ids.append([id if id != tokenizer.pad_token_id else -100 for id in ids])

    inputs.update({'labels': target_input_ids})
    return inputs

# Apply preprocessing to train and test datasets in batches
train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

# Initialize data collator for padding tokenized sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize T5 model for conditional generation
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Set up training arguments and directory for saving model outputs
TRAINING_OUTPUT = "./models/t5_fine_tuned_reviews"
training_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_strategy='epoch',
)

# Initialize Trainer object for training the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Perform model training
trainer.train()

# Save the fine-tuned model
trainer.save_model(TRAINING_OUTPUT)
# Set up GPU usage (Can be run on CPU too if GPU not available automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define function to generate reviews using the fine-tuned model
def generate_review(text):
    inputs = tokenizer("review: " + text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(inputs['input_ids'], max_length=128, no_repeat_ngram_size=3, num_beams=6, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Generate reviews for randomly selected products from the test dataset
random_products = test_dataset.shuffle(42).select(range(10))['product_title']

# Print generated reviews for sample products with different star ratings
print(generate_review(random_products[0] + ", 3 Stars!"))
print(generate_review(random_products[1] + ", 5 Stars!"))
print(generate_review(random_products[2] + ", 2 Stars!"))
