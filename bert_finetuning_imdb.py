# -*- coding: utf-8 -*-
"""Bert_finetuning_IMDB.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mRlThB8lzwG5UW8Mm4-QBpR2DQV4Q-a3
"""

# sentiment_pipeline.py

import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import logging
HF_TOKEN="Your key here"  # Replace with your Hugging Face token 
# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # yor wandb api key here

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main():
        # 1. Load the IMDb dataset
        logger.info("Loading dataset...")
        dataset = load_dataset("imdb")

        # 2. Load tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def preprocess(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

        # 3. Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(preprocess, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")

        # 4. Load pre-trained BERT model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # 5. Prepare training arguments
        training_args = TrainingArguments(
      output_dir="./results",
      do_train=True,
      do_eval=True,
      logging_steps=10,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=32,
      num_train_epochs=2,
      learning_rate=2e-5,
      weight_decay=0.01,
      save_steps=500,
      save_total_limit=2,
      logging_dir="./logs"
  )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # 6. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(10000)),  # Optional: use subset
            eval_dataset=tokenized_dataset["test"].select(range(2000)),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 7. Train the model
        logger.info("Training model...")
        trainer.train()

        # 8. Evaluate the model
        logger.info("Evaluating model...")
        metrics = trainer.evaluate()
        logger.info(f"Evaluation Metrics: {metrics}")

        # 9. Save model and tokenizer
        logger.info("Saving model...")
        model_dir = "./sentiment_model"
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        logger.info("Pipeline complete. Model saved.")

        # 10. Demonstrate loading and inference
        logger.info("Loading model for inference...")
        loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
        loaded_tokenizer = BertTokenizer.from_pretrained(model_dir)

        def predict_sentiment(text):
            inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            outputs = loaded_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            return "Positive" if pred == 1 else "Negative"

        # Test inference
        sample_text = "The movie was absolutely fantastic and I loved it!"
        prediction = predict_sentiment(sample_text)
        logger.info(f"Sample Text Prediction: {prediction}")


if __name__ == "__main__":
    main()

