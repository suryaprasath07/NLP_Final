import json
import re
import emoji
import pandas as pd
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np

# ------------------------------------------
# LOCAL MODEL PATH (HPC DIRECTORY)
# ------------------------------------------
LOCAL_MODEL_PATH = "/home/suryajk/scratch.hpcintro/train_transformer_h_files/distilbert_local"

def clean_text(text):
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags (keep the word)
    text = re.sub(r"#", "", text)
    # Remove non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_hatexplain(json_path="/home/suryajk/scratch.hpcintro/train_transformer_h_files/HateXplain/dataset.json", save_csv=True):
    with open(json_path) as f:
        data = json.load(f)
    
    def majority_label(annotators):
        labels = [a["label"] for a in annotators]
        return Counter(labels).most_common(1)[0][0]
    
    df = pd.DataFrame({
        "text": [" ".join(item["post_tokens"]) for item in data.values()],
        "label_name": [majority_label(item["annotators"]) for item in data.values()]
    })
    
    # Map to binary toxic labels
    mapping = {
        "hatespeech": "toxic",
        "offensive": "toxic",
        "normal": "non-toxic"
    }
    df["binary_label"] = df["label_name"].map(mapping)
    
    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Convert labels to 0/1
    df["label"] = df["binary_label"].apply(lambda x: 1 if x == "toxic" else 0)
    
    if save_csv:
        df[["clean_text", "label"]].to_csv("/home/suryajk/scratch.hpcintro/train_transformer_h_files/HateXplain/cleaned_dataset.csv", index=False)
    
    return df[["clean_text", "label"]]

def tokenize(batch):
    return tokenizer(batch["clean_text"], truncation=True, padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    """Compute accuracy for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def train_distilbert(df):
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    
    # ------------------------------------------
    # LOAD MODEL FROM LOCAL PATH (OFFLINE)
    # ------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        num_labels=2
    )
    
    training_args = TrainingArguments(
        output_dir="/home/suryajk/scratch.hpcintro/train_transformer_h_files/hate_model",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    metrics = trainer.evaluate(eval_dataset=test_ds)
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS:")
    print("="*50)
    print(f"Accuracy: {metrics['eval_accuracy']:.4f} ({metrics['eval_accuracy']*100:.2f}%)")
    print(f"Loss: {metrics['eval_loss']:.4f}")
    print("="*50 + "\n")
    
    trainer.save_model("/home/suryajk/scratch.hpcintro/train_transformer_h_files/hate_model")
    
    return metrics

def main():
    df = process_hatexplain()
    print("Sample processed row:")
    print(df.head())
    
    metrics = train_distilbert(df)
    
    # Save metrics to file
    with open("/home/suryajk/scratch.hpcintro/train_transformer_h_files/hate_model/metrics.txt", "w") as f:
        f.write(f"Final Accuracy: {metrics['eval_accuracy']:.4f}\n")
        f.write(f"Final Loss: {metrics['eval_loss']:.4f}\n")
    
    print("Training complete! Metrics saved.")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

if __name__ == "__main__":
    main()
