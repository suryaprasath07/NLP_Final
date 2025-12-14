from final_model import predict_toxicity

result = predict_toxicity("I hate you!")
print(result[0]['label'])

texts = [
    "Hello, nice to meet you!",
    "You're an idiot",
    "Great post, thanks for sharing!"
]

results = predict_toxicity(texts)

for result in results:
    print(f"{result['text']}: {result['label']} (confidence: {result['confidence']:.2f})")