from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re


app = FastAPI(
    title="AI Text Summarizer",
    description="Text summarization using a fine-tuned T5 model",
    version="1.0"
)


MODEL_PATH = "./final_model"


# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


# Select device
if torch.cuda.is_available():
    device = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")

else:
    device = torch.device("cpu")


model.to(device)
model.eval()


print("===================================")
print("AI Text Summarizer")
print("Model loaded successfully!")
print(f"Device: {device}")
print("===================================")


# Request body
class DialogueInput(BaseModel):
    dialogue: str


# Clean input text
def clean_data(text: str) -> str:

    text = re.sub(r"\r\n", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# Generate summary
def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# Summarization API
@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# Homepage
@app.get("/")
def home():

    return FileResponse(
        "templates/index.html"
    )