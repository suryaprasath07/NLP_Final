# In your other Python file (e.g., your Reddit API)
from final_model import predict_toxicity

# Example 1: Predict single text
result = predict_toxicity("I hate you!")
print(result[0]['label'])  # Returns: 'toxic' or 'non-toxic'

# Example 2: Predict multiple texts
texts = [
    "Hello, nice to meet you!",
    "You're an idiot",
    "Great post, thanks for sharing!"
]

results = predict_toxicity(texts)

for result in results:
    print(f"{result['text']}: {result['label']} (confidence: {result['confidence']:.2f})")