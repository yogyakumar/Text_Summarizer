# AI Text Summarizer

This is a text summarizer project that I made using Python, FastAPI and a fine-tuned T5-small model.

The main idea of this project is to take a long conversation or text and generate a shorter summary from it.

## What it can do

- Summarize long text and conversations
- Copy the generated summary
- Download the summary
- Try an example text
- Shows character and word count
- Works with a local trained model

## Technologies I used

- Python
- PyTorch
- Hugging Face Transformers
- T5-small
- FastAPI
- HTML
- CSS
- JavaScript

## How I made it

I first trained the T5-small model on the SAMSum dataset using Google Colab.

The dataset contains conversations along with their summaries. After training, I saved the trained model and used it in a FastAPI backend.

The frontend sends the text to the FastAPI API. The model processes it and returns the generated summary, which is then shown on the webpage.

## Project Structure

Text Summarizer Project/

├── app.py
├── requirements.txt
├── README.md
├── final_model/
└── templates/
    └── index.html

## Model Training

I used the SAMSum conversational dataset.

Training data:
- 14,732 examples

Validation data:
- 818 examples

Model:
- T5-small

Input length:
- 512 tokens

Maximum summary length:
- 150 tokens

The model was trained for 3 epochs using a T4 GPU in Google Colab.

## How to run it

First create a virtual environment:

```powershell
py -3.11 -m venv .venv