# src/evaluation/run_evaluation_coarse.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.load_model import get_models

DATA_PATH = Path("data/processed/reddit_for_eval.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CLASSES = ['joy', 'neutral', 'anger', 'surprise', 'sadness', 'fear']


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} examples")
    # keep only rows with valid coarse labels
    df = df[df['label_coarse'].isin(CLASSES)].reset_index(drop=True)

    models = get_models()
    results = []

    for name, predict_fn in models.items():
        print(f"\nRunning {name}")
        preds = []
        for text in tqdm(df['text'].tolist(), desc=name):
            try:
                lab = predict_fn(text)
            except Exception:
                lab = 'neutral'
            preds.append(lab)

        df[name] = preds
        macro_f1 = f1_score(df['label_coarse'], preds, average='macro', zero_division=0)
        report = classification_report(df['label_coarse'], preds, output_dict=True, zero_division=0)
        results.append({"model": name, "macro_f1": macro_f1, "report": report})

        # confusion matrix
        cm = confusion_matrix(df['label_coarse'], preds, labels=CLASSES)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f"Confusion Matrix – {name}")
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_{name[:30].replace('/','_')}.png", dpi=200)
        plt.close()

    comp = pd.DataFrame(results)[['model','macro_f1']].sort_values('macro_f1', ascending=False)
    comp.to_csv(RESULTS_DIR / 'model_comparison_coarse.csv', index=False)
    print('\nAll done! Results saved to results/')
    print(comp)

if __name__ == '__main__':
    main()

