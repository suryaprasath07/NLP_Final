# src/models/train_distilbert_coarse.py
# src/models/train_distilbert_coarse.py
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
AutoTokenizer, AutoModelForSequenceClassification,
TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import torch
from transformers import default_data_collator

def preprocess(batch):
    enc = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    enc['labels'] = batch['label']
    return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    micro = f1_score(labels, preds, average='micro', zero_division=0)
    report = classification_report(labels, preds, target_names=COARSE_LABELS, zero_division=0, output_dict=True)
    return {"macro_f1": macro, "micro_f1": micro, **{f"f1_{k}": v['f1-score'] for k,v in report.items() if k in COARSE_LABELS}}

CSV_PATH = Path("data/processed/reddit_for_eval.csv")
OUT_DIR = Path("goemo_coarse")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6
EPOCHS = 3
BATCH_SIZE = 16
LR = 3e-5
MAX_LENGTH = 128
SEED = 42

COARSE_LABELS = ["joy","neutral","anger","surprise","sadness","fear"]
LABEL2ID = {l:i for i,l in enumerate(COARSE_LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}


print("Loading processed CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
assert 'text' in df.columns and 'label_coarse' in df.columns, "CSV must have 'text' and 'label_coarse' columns"
df = df[df['label_coarse'].notna()].copy()
df = df[df['label_coarse'].isin(COARSE_LABELS)].reset_index(drop=True)
print(f"Dataset rows after filtering: {len(df):,}")

df['label_id'] = df['label_coarse'].map(LABEL2ID)

train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['label_id'])
train_df, val_df = train_test_split(train_df, test_size=0.12, random_state=SEED, stratify=train_df['label_id'])
print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

ds_train = Dataset.from_pandas(train_df[['text','label_id']].rename(columns={'label_id':'label'}))
ds_val = Dataset.from_pandas(val_df[['text','label_id']].rename(columns={'label_id':'label'}))
ds_test = Dataset.from_pandas(test_df[['text','label_id']].rename(columns={'label_id':'label'}))


dataset = DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})

print("Loading tokenizer and tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

dataset = dataset.map(preprocess, batched=True, remove_columns=['text'])


print("Loading model")
model = AutoModelForSequenceClassification.from_pretrained(
MODEL_NAME,
num_labels=NUM_LABELS,
problem_type="single_label_classification"
)

# -------------------------
# Training arguments + Trainer
# -------------------------
training_args = TrainingArguments(
    output_dir=str(OUT_DIR / "ckpts"),
    eval_strategy="epoch",
    save_strategy="epoch",

    num_train_epochs=5,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=2e-5,

    weight_decay=0.01,
    logging_steps=200,
    
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,

    seed=SEED,
    save_total_limit=2,
    report_to=[],
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)


print("Starting training...")
trainer.train()

print("Saving model and label map to:", OUT_DIR)
trainer.save_model(str(OUT_DIR / "model"))
tokenizer.save_pretrained(str(OUT_DIR / "model"))
with open(OUT_DIR / "model" / "label_map.json", 'w') as f:
    json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID, "labels": COARSE_LABELS}, f, indent=2)
