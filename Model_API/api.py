from flask import Flask, jsonify, request
from flask_cors import CORS

import sys
import os
sys.path.append(os.path.abspath("../"))

from Hate_Detection_Model.final_model.predict_toxicity import predict_toxicity
from Emotion_Classification_Model.src.models.final_model.final_prediction import predict_emotion

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests


@app.route('/api/predict/combined', methods=['POST'])
def predict_combined_api():
    """
    Predict both toxicity and emotion for an array of texts.
    
    Expected JSON body:
    {
        "texts": ["text1", "text2", "text3", ...]
    }
    
    Returns:
    {
        "success": true,
        "count": 3,
        "predictions": [
            {
                "text": "text1",
                "toxicity": {
                    "label": "toxic",
                    "confidence": 0.95,
                    "toxic_probability": 0.95,
                    "non_toxic_probability": 0.05
                },
                "emotion": {
                    "label": "anger",
                    "confidence": 0.87,
                    "distribution": {
                        "joy": 0.05,
                        "neutral": 0.03,
                        "anger": 0.87,
                        "surprise": 0.02,
                        "sadness": 0.02,
                        "fear": 0.01
                    }
                }
            },
            ...
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        # Validate input
        if not isinstance(texts, list):
            return jsonify({
                "success": False,
                "error": "'texts' must be an array"
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                "success": False,
                "error": "'texts' array cannot be empty"
            }), 400
        
        # Limit to prevent abuse
        if len(texts) > 1000:
            return jsonify({
                "success": False,
                "error": "Maximum 1000 texts allowed per request"
            }), 400
        
        # Get predictions from both models
        toxicity_predictions = predict_toxicity(texts)
        emotion_predictions = predict_emotion(texts)
        
        # Combine results
        combined_predictions = []
        for tox_pred, emo_pred in zip(toxicity_predictions, emotion_predictions):
            combined_predictions.append({
                "text": tox_pred['text'],
                "toxicity": {
                    "label": tox_pred['label'],
                    "confidence": tox_pred['confidence'],
                    "toxic_probability": tox_pred['toxic_probability'],
                    "non_toxic_probability": tox_pred['non_toxic_probability']
                },
                "emotion": {
                    "label": emo_pred['emotion'],
                    "confidence": emo_pred['confidence'],
                    "distribution": emo_pred['distribution']
                }
            })
        
        return jsonify({
            "success": True,
            "count": len(combined_predictions),
            "predictions": combined_predictions
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict/combined/single', methods=['POST'])
def predict_combined_single():
    """
    Predict both toxicity and emotion for a single text.
    
    Expected JSON body:
    {
        "text": "Your text here"
    }
    
    Returns:
    {
        "success": true,
        "prediction": {
            "text": "Your text here",
            "toxicity": {
                "label": "toxic",
                "confidence": 0.95,
                "toxic_probability": 0.95,
                "non_toxic_probability": 0.05
            },
            "emotion": {
                "label": "anger",
                "confidence": 0.87,
                "distribution": {
                    "joy": 0.05,
                    "neutral": 0.03,
                    "anger": 0.87,
                    "surprise": 0.02,
                    "sadness": 0.02,
                    "fear": 0.01
                }
            }
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "success": False,
                "error": "'text' must be a non-empty string"
            }), 400
        
        # Get predictions from both models
        toxicity_predictions = predict_toxicity([text])
        emotion_predictions = predict_emotion([text])
        
        tox_pred = toxicity_predictions[0]
        emo_pred = emotion_predictions[0]
        
        # Combine results
        combined_prediction = {
            "text": tox_pred['text'],
            "toxicity": {
                "label": tox_pred['label'],
                "confidence": tox_pred['confidence'],
                "toxic_probability": tox_pred['toxic_probability'],
                "non_toxic_probability": tox_pred['non_toxic_probability']
            },
            "emotion": {
                "label": emo_pred['emotion'],
                "confidence": emo_pred['confidence'],
                "distribution": emo_pred['distribution']
            }
        }
        
        return jsonify({
            "success": True,
            "prediction": combined_prediction
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict/toxicity', methods=['POST'])
def predict_toxicity_only():
    """
    Predict only toxicity for an array of texts.
    
    Expected JSON body:
    {
        "texts": ["text1", "text2", ...]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "success": False,
                "error": "'texts' must be a non-empty array"
            }), 400
        
        if len(texts) > 1000:
            return jsonify({
                "success": False,
                "error": "Maximum 1000 texts allowed per request"
            }), 400
        
        predictions = predict_toxicity(texts)
        
        return jsonify({
            "success": True,
            "count": len(predictions),
            "predictions": predictions
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict/emotion', methods=['POST'])
def predict_emotion_only():
    """
    Predict only emotion for an array of texts.
    
    Expected JSON body:
    {
        "texts": ["text1", "text2", ...]
    }
    
    Returns:
    {
        "success": true,
        "count": 2,
        "predictions": [
            {
                "text": "text1",
                "emotion": "joy",
                "confidence": 0.92
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "success": False,
                "error": "'texts' must be a non-empty array"
            }), 400
        
        if len(texts) > 1000:
            return jsonify({
                "success": False,
                "error": "Maximum 1000 texts allowed per request"
            }), 400
        
        predictions = predict_emotion(texts)
        
        return jsonify({
            "success": True,
            "count": len(predictions),
            "predictions": predictions
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Text Analysis API - Toxicity & Emotion Detection"
    })


@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "message": "Text Analysis API - Toxicity & Emotion Detection",
        "version": "1.0",
        "endpoints": {
            "/api/health": {
                "method": "GET",
                "description": "Health check"
            },
            "/api/predict/combined": {
                "method": "POST",
                "description": "Predict both toxicity and emotion for multiple texts",
                "body": {"texts": ["text1", "text2"]}
            },
            "/api/predict/combined/single": {
                "method": "POST",
                "description": "Predict both toxicity and emotion for a single text",
                "body": {"text": "Your text here"}
            },
            "/api/predict/toxicity": {
                "method": "POST",
                "description": "Predict only toxicity for multiple texts",
                "body": {"texts": ["text1", "text2"]}
            },
            "/api/predict/emotion": {
                "method": "POST",
                "description": "Predict only emotion for multiple texts",
                "body": {"texts": ["text1", "text2"]}
            }
        },
        "example_usage": {
            "curl_combined": 'curl -X POST http://localhost:8001/api/predict/combined -H "Content-Type: application/json" -d \'{"texts": ["I hate this", "I love this"]}\'',
            "curl_single": 'curl -X POST http://localhost:8001/api/predict/combined/single -H "Content-Type: application/json" -d \'{"text": "I am so happy!"}\''
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)