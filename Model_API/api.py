from flask import Flask, jsonify, request
from flask_cors import CORS

import sys
import os
sys.path.append(os.path.abspath("../"))

from Hate_Detection_Model.final_model.predict_toxicity import predict_toxicity

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/api/predict/toxicity', methods=['POST'])
def predict_toxicity_api():
    """
    Predict toxicity for an array of texts.
    
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
                "label": "toxic",
                "confidence": 0.95,
                "toxic_probability": 0.95,
                "non_toxic_probability": 0.05
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
        
        # Get predictions
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

@app.route('/api/predict/toxicity/single', methods=['POST'])
def predict_single_text():
    """
    Predict toxicity for a single text.
    
    Expected JSON body:
    {
        "text": "Your text here"
    }
    
    Returns:
    {
        "success": true,
        "prediction": {
            "text": "Your text here",
            "label": "toxic",
            "confidence": 0.95,
            "toxic_probability": 0.95,
            "non_toxic_probability": 0.05
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
        
        # Get prediction
        predictions = predict_toxicity([text])
        
        return jsonify({
            "success": True,
            "prediction": predictions[0]
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
        "service": "Toxicity Detection API"
    })

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "message": "Toxicity Detection API",
        "endpoints": {
            "/api/health": "Health check",
            "http://localhost:8001/api/predict/toxicity": "POST - Predict toxicity for multiple texts (body: {texts: []})",
            "/api/predict/toxicity/single": "POST - Predict toxicity for single text (body: {text: ''})"
        }
    })

if __name__ == '__main__':
    print("Starting Hate Detection API...")
    app.run(debug=True, host='0.0.0.0', port=8001)