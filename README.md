# Sentiment Analysis Pipeline using BERT (Fine-tuned on IMDB)

## Overview

This project implements a sentiment analysis pipeline using a pre-trained BERT model fine-tuned on the IMDB movie review dataset. The pipeline performs binary classification (positive/negative sentiment) using Hugging Face Transformers. It includes data loading, preprocessing with `BertTokenizer`, model training using `Trainer`, and final model deployment via Hugging Face Model Hub.

### Pipeline Components and Design Rationale

The pipeline uses `bert-base-uncased` as the backbone due to its proven performance on NLP tasks with minimal preprocessing. Tokenization is handled using Hugging Face’s `AutoTokenizer`, ensuring compatibility with the base model. The dataset is loaded via `datasets.load_dataset`, which simplifies access to IMDB and supports automatic train/test splits. 

The model is fine-tuned using the `Trainer` API for efficient training and evaluation, with metrics like accuracy and F1-score. This high-level interface abstracts away boilerplate training code and allows easy experimentation.

### Challenges and Solutions

Anticipated challenges include high memory requirements for BERT fine-tuning and managing long sequences. These were addressed by truncating sequences and using batch sizes that fit within Colab’s memory constraints. Logging was disabled or configured locally to avoid integration errors with third-party tools like Weights & Biases. Model checkpoints are saved locally, then pushed to Hugging Face manually using `push_to_hub()`.

---

## Files

- `sentiment_pipeline.py`: Python script containing the entire pipeline.
- `sentiment-model/`: Directory with fine-tuned model and tokenizer files.
- `README.md`: This file.

## Hugging Face Model Link

👉 [View the Fine-Tuned Model on Hugging Face](https://huggingface.co/HrishikeshDeore/Bert_base_finetuned_IMDB) 

---

## Requirements

- `transformers`
- `datasets`
- `sklearn`
- `torch`
- `huggingface_hub`
