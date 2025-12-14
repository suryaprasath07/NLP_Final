import torch
import re
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(CURRENT_DIR, "..", "train_transformer_h_files", "checkpoint-20160")
CHECKPOINT_PATH = os.path.abspath(MODEL_DIR)

print(f"[INFO] Looking for model at: {CHECKPOINT_PATH}")

def clean_text(text):
    """Clean text the same way as during training"""
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

def load_model(checkpoint_path):
    """Load the trained model and tokenizer"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer, device

def predict_text(text, model, tokenizer, device):
    """Predict if a single text is toxic or not"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    # Get confidence scores
    confidence = probabilities[0][prediction].item()
    toxic_prob = probabilities[0][1].item()
    non_toxic_prob = probabilities[0][0].item()
    
    label = "toxic" if prediction == 1 else "non-toxic"
    
    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "label": label,
        "confidence": confidence,
        "toxic_probability": toxic_prob,
        "non_toxic_probability": non_toxic_prob
    }

def predict_batch(texts, model, tokenizer, device, batch_size=16):
    """Predict multiple texts efficiently"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        cleaned_texts = [clean_text(t) for t in batch_texts]
        
        # Tokenize batch
        inputs = tokenizer(
            cleaned_texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # Process results
        for j, (text, cleaned, pred, probs) in enumerate(
            zip(batch_texts, cleaned_texts, predictions, probabilities)
        ):
            label = "toxic" if pred.item() == 1 else "non-toxic"
            results.append({
                "text": text,
                "cleaned_text": cleaned,
                "label": label,
                "confidence": probs[pred].item(),
                "toxic_probability": probs[1].item(),
                "non_toxic_probability": probs[0].item()
            })
    
    return results

_model = None
_tokenizer = None
_device = None

def get_model():
    """Lazy load the model (loads only once, then cached)"""
    global _model, _tokenizer, _device
    
    if _model is None:
        print("Loading model for the first time...")
        _model, _tokenizer, _device = load_model(CHECKPOINT_PATH)
    
    return _model, _tokenizer, _device

def predict_toxicity(texts, batch_size=16):
    """
    Main function to predict toxicity for an array of texts.
    Can be called directly from other files.
    
    Args:
        texts (list or str): Single text string or list of text strings
        batch_size (int): Batch size for processing (default: 16)
    
    Returns:
        list: List of dictionaries containing predictions for each text
              Each dict contains: text, label, confidence, toxic_probability, non_toxic_probability
    
    Example:
        from hate_speech_detector import predict_toxicity
        
        results = predict_toxicity(["Hello world", "I hate you"])
        for result in results:
            print(f"{result['text']}: {result['label']} ({result['confidence']:.2f})")
    """
    if isinstance(texts, str):
        texts = [texts]
    
    model, tokenizer, device = get_model()
    results = predict_batch(texts, model, tokenizer, device, batch_size)
    
    return results