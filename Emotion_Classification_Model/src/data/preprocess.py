# preprocess.py
import pandas as pd
import re
import json
from pathlib import Path
import emoji
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 27 GoEmotions fine-grained labels → 6 coarse labels (standard mapping
# ------------------------------------------------------------------
GOEMOTIONS_TO_COARSE = {
    "admiration":     "joy",
    "amusement":      "joy",
    "approval":       "joy",
    "caring":         "joy",
    "excitement":     "joy",
    "gratitude":      "joy",
    "joy":            "joy",
    "love":           "joy",
    "optimism":       "joy",
    "pride":          "joy",
    "relief":         "joy",

    "anger":          "anger",
    "annoyance":      "anger",
    "disapproval":    "anger",
    "disgust":        "anger",

    "fear":           "fear",
    "nervousness":    "fear",

    "sadness":        "sadness",
    "disappointment": "sadness",
    "embarrassment":  "sadness",
    "grief":          "sadness",
    "remorse":        "sadness",

    "surprise":       "surprise",
    "confusion":      "surprise",
    "curiosity":      "surprise",
    "realization":   "surprise",

    "desire":         "neutral",
    "neutral":        "neutral",
}

GOEMO_LABELS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise'
]

def map_to_coarse_safe(raw_label):
    """
    Input : anything the dataset might give us (int, str, list, [], None, '[0, 17]', etc.)
    Output: one of 'joy', 'anger', 'sadness', 'fear', 'surprise', 'neutral'
    """
    if pd.isna(raw_label) or raw_label in (None, "", [], {}):
        return "neutral"

    label_ids = []

    if isinstance(raw_label, (int, float)):
        if not pd.isna(raw_label):
            label_ids = [int(raw_label)]
    elif isinstance(raw_label, str):
        raw_label = raw_label.strip()
        if raw_label in ("[]", "", "nan", "None"):
            return "neutral"
        found = re.findall(r'\d+', raw_label)
        label_ids = [int(x) for x in found]
    else:
        try:
            label_ids = [int(x) for x in raw_label if str(x).isdigit()]
        except:
            pass

    if label_ids:
        for idx in label_ids:
            if 0 <= idx < len(GOEMO_LABELS):
                fine_name = GOEMO_LABELS[idx]
                if fine_name in GOEMOTIONS_TO_COARSE:
                    return GOEMOTIONS_TO_COARSE[fine_name]

    # Fallback --> neutral
    return "neutral"


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------
def demojize_text(s):
    return emoji.demojize(str(s), language='en') if s else ""

def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r'http\S+|www\.\S+', '', s)          # URLs
    s = re.sub(r'\s+', ' ', s)                      # multiple spaces
    s = demojize_text(s)
    return s.strip()


def detect_text_column(df):
    candidates = ['text', 'comment', 'body', 'sentence', 'utterance', 'content']
    for c in df.columns:
        if str(c).lower() in candidates:
            return c
    lengths = {c: df[c].astype(str).str.len().median() for c in df.columns}
    return max(lengths, key=lengths.get)

def detect_label_column(df):
    candidates = ['labels', 'label', 'emotions', 'emotion', 'label_text']
    for c in df.columns:
        if str(c).lower() in candidates:
            return c
    return None


def load_and_process_csv(path: Path):
    df = pd.read_csv(path)

    text_col = detect_text_column(df)
    label_col = detect_label_column(df)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {path.name}"):
        text = clean_text(row[text_col])

        raw_label = row[label_col] if label_col and label_col in row else None
        coarse = map_to_coarse_safe(raw_label)

        rows.append({
            "text": text,
            "label_coarse": coarse,
            "label_raw": str(raw_label),
            "source_file": path.name,
            "meta": json.dumps({
                k: str(row[k]) for k in ["id", "author", "subreddit", "score", "created_utc"]
                if k in row and pd.notna(row[k])
            })
        })

    return pd.DataFrame(rows)


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in data/raw/. Run download_datasets.py first.")
        return

    all_frames = []
    for p in csv_files:
        try:
            df_processed = load_and_process_csv(p)
            print(f"Processed {p.name}: {len(df_processed)} rows")
            all_frames.append(df_processed)
        except Exception as e:
            print(f"Failed on {p.name}: {e}")

    final_df = pd.concat(all_frames, ignore_index=True)
    out_path = OUT_DIR / "reddit_for_eval.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(final_df):,} rows → {out_path}")

    print("\nCoarse label distribution:")
    print(final_df["label_coarse"].value_counts())

if __name__ == "__main__":
    main()