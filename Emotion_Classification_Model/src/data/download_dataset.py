from datasets import load_dataset, DatasetDict
import os
import json
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_dataset_to_csv(ds, outpath):
    ds.to_pandas().to_csv(outpath, index=False)
    print(f"Saved {outpath} rows: {len(ds)}")

def main():
    print("1) Attempting to load `reddit_emotions` (pre-labeled reddit emotion dataset)...")
    try:
        d_re = load_dataset("reddit_emotions")
        print("Loaded reddit_emotions with splits:", d_re.keys())
        for split_name, ds in d_re.items():
            outp = OUT_DIR / f"reddit_emotions_{split_name}.csv"
            save_dataset_to_csv(ds, outp)
    except Exception as e:
        print("Could not load reddit_emotions:", e)

    print("\n2) Attempting to load GoEmotions Reddit subset or go_emotions generally...")
    tried = []
    for dataset_id in ["tahirbalarabe/go_emotions_reddit", "go_emotions", "go_emotions_dataset"]:
        try:
            ds = load_dataset(dataset_id)
            print(f"Loaded {dataset_id} with splits: {ds.keys()}")
            for sp, dset in ds.items():
                outp = OUT_DIR / f"{dataset_id.replace('/', '_')}_{sp}.csv"
                save_dataset_to_csv(dset, outp)
            tried.append(dataset_id)
            break
        except Exception as e:
            print(f"Could not load {dataset_id}: {e}")
    if not tried:
        print("No GoEmotions variant found automatically. You can still proceed with reddit_emotions if present.")

    print("\n3) Optionally: Save a combined sample file for quick experiments (sample up to N)")
    try:
        ds = load_dataset("reddit_emotions")
        combined = ds["train"].train_test_split(test_size=0.2, seed=42)
        sample = combined["test"].select(range(min(5000, len(combined["test"]))))
        outp = OUT_DIR / "reddit_emotions_sample_5k.csv"
        save_dataset_to_csv(sample, outp)
        print("Saved sample for quick testing:", outp)
    except Exception:
        print("Skipping reddit_emotions sampling (not available).")

    print("\nDone. Raw CSV files are in:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
