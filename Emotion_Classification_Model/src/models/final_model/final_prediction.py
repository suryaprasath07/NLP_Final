import os
import re
from pathlib import Path
from transformers import pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "goemo_coarse", "model")
MODEL_PATH = os.path.abspath(MODEL_DIR)

COARSE_LABELS = ["joy", "neutral", "anger", "surprise", "sadness", "fear"]

print(f"[INFO] Looking for emotion model at: {MODEL_PATH}")


def clean_text(text):
    """Clean text before prediction"""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags (keep the word)
    text = re.sub(r"#", "", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model(model_path):
    """Load the trained emotion classification model"""
    print(f"Loading emotion model from {model_path}...")
    
    # Load model using pipeline with all scores
    pipe = pipeline(
        "text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        return_all_scores=True,  # Changed to True to get all emotion probabilities
        device=-1
    )
    
    print("Emotion model loaded successfully")
    return pipe


def parse_emotion_scores(outputs):
    """Parse model outputs to get emotion distribution"""
    emotion_dist = {label: 0.0 for label in COARSE_LABELS}
    predicted_emotion = 'neutral'
    max_score = 0.0
    
    if isinstance(outputs, list) and len(outputs) > 0:
        for item in outputs:
            if isinstance(item, dict) and 'label' in item and 'score' in item:
                lab = str(item['label']).lower()
                score = float(item['score'])
                
                # Try to match emotion labels
                matched = False
                for c in COARSE_LABELS:
                    if c in lab:
                        emotion_dist[c] = score
                        matched = True
                        if score > max_score:
                            max_score = score
                            predicted_emotion = c
                        break
                
                # Handle label_N format
                if not matched and lab.startswith('label_'):
                    try:
                        idx = int(lab.split('_')[-1])
                        if 0 <= idx < len(COARSE_LABELS):
                            emotion_label = COARSE_LABELS[idx]
                            emotion_dist[emotion_label] = score
                            if score > max_score:
                                max_score = score
                                predicted_emotion = emotion_label
                    except:
                        pass
    
    return predicted_emotion, max_score, emotion_dist


def predict_single(text, pipe):
    """Predict emotion for a single text with distribution"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Get prediction with all scores
    outputs = pipe(cleaned_text)
    
    # Parse outputs - handle nested list structure
    if isinstance(outputs, list) and len(outputs) > 0:
        scores = outputs[0] if isinstance(outputs[0], list) else outputs
    else:
        scores = outputs
    
    emotion, confidence, distribution = parse_emotion_scores(scores)
    
    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "emotion": emotion,
        "confidence": confidence,
        "distribution": distribution
    }


def predict_batch_texts(texts, pipe, batch_size=16):
    """Predict emotions for multiple texts efficiently with distributions"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        cleaned_texts = [clean_text(t) for t in batch_texts]
        
        # Get predictions for batch with all scores
        batch_outputs = pipe(cleaned_texts)
        
        # Process results
        for text, cleaned, outputs in zip(batch_texts, cleaned_texts, batch_outputs):
            emotion, confidence, distribution = parse_emotion_scores(outputs)
            
            results.append({
                "text": text,
                "cleaned_text": cleaned,
                "emotion": emotion,
                "confidence": confidence,
                "distribution": distribution
            })
    
    return results


# Global variables for lazy loading
_emotion_model = None


def get_emotion_model():
    """Lazy load the model (loads only once, then cached)"""
    global _emotion_model
    
    if _emotion_model is None:
        print("Loading emotion model for the first time...")
        _emotion_model = load_model(MODEL_PATH)
    
    return _emotion_model


def predict_emotion(texts, batch_size=16):
    """
    Main function to predict emotions for an array of texts with distribution.
    Can be called directly from other files for API endpoints.
    
    Args:
        texts (list or str): Single text string or list of text strings
        batch_size (int): Batch size for processing (default: 16)
    
    Returns:
        list: List of dictionaries containing predictions for each text
              Each dict contains: text, cleaned_text, emotion, confidence, distribution
    
    Example:
        from emotion_classifier import predict_emotion
        
        # Single text
        result = predict_emotion("I am so happy today!")
        print(result[0]['emotion'])  # Output: 'joy'
        print(result[0]['distribution'])  # Output: {'joy': 0.95, 'neutral': 0.02, ...}
        
        # Multiple texts
        results = predict_emotion([
            "I am so happy today!",
            "This makes me angry",
            "I'm feeling scared"
        ])
        for result in results:
            print(f"{result['text']}: {result['emotion']} ({result['confidence']:.2f})")
            print(f"Distribution: {result['distribution']}")
    """
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
    
    pipe = get_emotion_model()
    
    # Get predictions
    results = predict_batch_texts(texts, pipe, batch_size)
    
    return results


def predict_emotion_simple(text):
    """
    Simplified function that returns just the emotion label for a single text.
    
    Args:
        text (str): Input text string
    
    Returns:
        str: Predicted emotion label (one of: joy, neutral, anger, surprise, sadness, fear)
    
    Example:
        from emotion_classifier import predict_emotion_simple
        
        emotion = predict_emotion_simple("I am so happy!")
        print(emotion)  # Output: 'joy'
    """
    result = predict_emotion(text)
    return result[0]['emotion'] if result else 'neutral'


# For testing
if __name__ == "__main__":
    # Test single prediction
    print("\n=== Testing Single Prediction with Distribution ===")
    test_text = "I am so happy and excited today!"
    result = predict_emotion(test_text)
    print(f"Text: {result[0]['text']}")
    print(f"Emotion: {result[0]['emotion']}")
    print(f"Confidence: {result[0]['confidence']:.4f}")
    print(f"Distribution:")
    for emotion, score in result[0]['distribution'].items():
        print(f"  {emotion:10s}: {score:.4f} ({score*100:.2f}%)")
    
    # Test batch prediction
    print("\n=== Testing Batch Prediction ===")
    test_texts = [
        "I am so happy today!",
        "This makes me really angry",
        "I'm feeling very sad",
        "That was surprising!",
        "I'm scared of this",
        "Just a normal day"
    ]
    
    results = predict_emotion(test_texts)
    for result in results:
        print(f"\n{result['emotion']:10s} ({result['confidence']:.2f}) - {result['text']}")
        top_3 = sorted(result['distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3: {', '.join([f'{e}: {s:.2f}' for e, s in top_3])}")