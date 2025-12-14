from transformers import pipeline
from pathlib import Path
import numpy as np

COARSE_LABELS = ["joy","neutral","anger","surprise","sadness","fear"]

def get_models():
    models = {}

    model_path = Path("goemo_coarse/model")
    if model_path.exists():
        pipe = pipeline("text-classification", model=str(model_path), tokenizer=str(model_path), return_all_scores=False, device=-1)
        def predict_coarse(text):
            out = pipe(text)
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and 'label' in out[0]:
                lab = out[0]['label']
                lab = str(lab).lower()
                for c in COARSE_LABELS:
                    if c in lab:
                        return c
                if lab.startswith('label_'):
                    idx = int(lab.split('_')[-1])
                    return COARSE_LABELS[idx] if 0 <= idx < len(COARSE_LABELS) else 'neutral'
            return 'neutral'
        models['our_coarse_distilbert'] = predict_coarse

    try:
        pipe_jh = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)
        def wrap_jh(text):
            out = pipe_jh(text)
            lab = out[0]['label'].lower()
            if 'joy' in lab or 'amuse' in lab or 'love' in lab:
                return 'joy'
            if 'anger' in lab or 'annoy' in lab or 'disgust' in lab:
                return 'anger'
            if 'sad' in lab or 'grief' in lab or 'remorse' in lab:
                return 'sadness'
            if 'fear' in lab or 'nerv' in lab:
                return 'fear'
            if 'surpris' in lab or 'shock' in lab:
                return 'surprise'
            return 'neutral'
        models['distilroberta'] = wrap_jh
    except Exception:
        pass

    try:
        pipe_ds = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=-1)
        def wrap_ds(text):
            out = pipe_ds(text)
            lab = out[0]['label'].lower()
            if 'joy' in lab or 'happy' in lab:
                return 'joy'
            if 'anger' in lab:
                return 'anger'
            if 'sad' in lab:
                return 'sadness'
            if 'fear' in lab:
                return 'fear'
            if 'surpris' in lab:
                return 'surprise'
            return 'neutral'
        models['distilbert-emotion'] = wrap_ds
    except Exception:
        pass

    try:
        pipe_m = pipeline("text-classification", model="michellejieli/emotion_text_classifier", device=-1)
        def wrap_m(text):
            out = pipe_m(text)
            lab = out[0]['label'].lower()
            if 'joy' in lab or 'amuse' in lab:
                return 'joy'
            if 'anger' in lab:
                return 'anger'
            if 'sad' in lab:
                return 'sadness'
            if 'fear' in lab:
                return 'fear'
            if 'surpris' in lab:
                return 'surprise'
            return 'neutral'
        models['michelle'] = wrap_m
    except Exception:
        pass

    try:
        pipe_ro = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", function_to_apply="sigmoid", device=-1, top_k=None)
        def wrap_ro(text):
            out = pipe_ro(text)[0]
            score_map = {d['label'].lower(): float(d['score']) for d in out}
            joy_keys = ['admiration','amusement','approval','caring','excitement','gratitude','joy','love','optimism','pride','relief']
            anger_keys = ['anger','annoyance','disapproval','disgust']
            sadness_keys = ['sadness','disappointment','embarrassment','grief','remorse']
            fear_keys = ['fear','nervousness']
            surprise_keys = ['surprise','confusion','curiosity','realization']
            scores = {k:0.0 for k in COARSE_LABELS}
            for k,v in score_map.items():
                if k in joy_keys:
                    scores['joy'] += v
                elif k in anger_keys:
                    scores['anger'] += v
                elif k in sadness_keys:
                    scores['sadness'] += v
                elif k in fear_keys:
                    scores['fear'] += v
                elif k in surprise_keys:
                    scores['surprise'] += v
                else:
                    scores['neutral'] += v
            return max(scores, key=scores.get)
        models['roberta-goemotions'] = wrap_ro
    except Exception:
        pass

    return models
