# Reddit Hate Speech Detection System

## Overview

This project provides an end-to-end solution for detecting hate speech and toxic content on Reddit. It combines a fine-tuned DistilBERT model with a web-based interface that allows users to analyze posts and comments from any subreddit in real-time.

### Why This Project?

Social media platforms like Reddit host millions of discussions daily, but moderating harmful content at scale remains a significant challenge. This system:

- **Automates hate speech detection** to assist content moderators
- **Provides real-time analysis** of Reddit posts and comments
- **Offers transparency** through confidence scores and probability metrics
- **Enables research** into online toxicity patterns across communities

---

## What Was Done?

### 1. Model Fine-Tuning

A **DistilBERT-base-uncased** model was fine-tuned using the [HateXplain dataset](https://github.com/hate-alert/HateXplain/tree/master/Data/dataset.json) to classify text as toxic or non-toxic. The model was specifically optimized to detect hate speech patterns commonly found on Reddit.

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Dataset**: HateXplain (20,000+ human-annotated examples)
- **Task**: Binary classification (toxic/non-toxic)
- **Training Environment**: University of Maryland High-Performance Computing Cluster (HPCC)

### 2. HPC-Based Training

The model training was conducted on the [HPCC at UMD](https://hpcc.umd.edu/), utilizing GPU acceleration for efficient fine-tuning. All code and file paths were configured for this HPC environment.

### 3. Training Code

**Location**: `Hate_Detection_Model/main.py`

**Use this file if you want to:**
- Fine-tune the model from scratch
- Experiment with different hyperparameters
- Train on custom datasets

**Requirements**: Install dependencies from `Hate_Detection_Model/train_requirements.txt`

**Important Note**: This code was designed for HPC environments with GPU support. Some CUDA-related packages (like `torch` with CUDA) may not install correctly on standard consumer machines without dedicated NVIDIA GPUs. If you encounter installation issues on a local machine, you may need to install CPU-only versions of PyTorch or skip GPU-dependent packages.

### 4. Model Inference

**Location**: `Hate_Detection_Model/final_model/`

**Use this directory if you want to:**
- Load the pre-trained model for predictions
- Integrate the model into your own applications
- Run inference without re-training

### 5. Dataset Collection

**Location**: `dataset/` folder

Contains all scripts for:
- Scraping Reddit posts and comments
- Preprocessing text data
- Building the Reddit API interface

---

## How to Run the Project

### Prerequisites

- Python 3.8 or higher
- Git
- Internet connection (for Reddit API access)

### Step 1: Clone the Repository
```bash
git clone git@github.com:suryaprasath07/NLP_Final.git
cd NLP_Final
```

### Step 2: Start the Frontend

The frontend is a static HTML/CSS/JavaScript application with no dependencies.
```bash
python3 -m http.server 3000
```

The interface will be available at: `http://localhost:3000`

### Step 3: Start the Model API

The Model API handles toxicity predictions using the fine-tuned DistilBERT model.
```bash
# Create and activate virtual environment
python3 -m venv model-venv
source model-venv/bin/activate  # On Windows: model-venv\Scripts\activate

# Install dependencies
pip install -r Hate_Detection_Model/final_model/final_requirements.txt

# Start the API
python Model_API/api.py
```

The Model API will run on: `http://localhost:8001`

### Step 4: Start the Reddit Data API

The Data API fetches posts and comments from Reddit.
```bash
# Create and activate virtual environment (in a new terminal)
python3 -m venv data-venv
source data-venv/bin/activate  # On Windows: data-venv\Scripts\activate

# Install dependencies
pip install -r dataset/data_requirements.txt

# Start the API
python dataset/api.py
```

The Data API will run on: `http://localhost:8000`

### Step 5: Use the Application

1. Open your browser and navigate to `http://localhost:3000`
2. Enter a subreddit name (e.g., "science", "UMD", "gaming")
3. Click "Search" to fetch and analyze posts
4. Hover over any text to see toxicity predictions with confidence scores
5. Click "Load More Posts" to fetch additional content

---

## Project Structure
```
reddit-hate-speech-detection/
│
├── Hate_Detection_Model/
│   ├── main.py                          # Model training script
│   ├── train_requirements.txt           # Dependencies for training
│   │
│   ├── final_model/
│   │   ├── predict_toxicity.py          # Model inference logic
│   │   └── final_requirements.txt       # Dependencies for inference
│   │
│   └── train_transformer_h_files/
│       └── checkpoint-20160/            # Fine-tuned model weights
│
├── Model_API/
│   └── api.py                           # Toxicity detection API
│
├── dataset/
│   ├── api.py                           # Reddit scraping API
│   ├── data_requirements.txt            # Dependencies for data collection
│   └── scraper.py                       # Reddit data collection logic
│
├── index.html                           # Frontend interface
└── README.md                            # This file
```

---

## Features

- **Real-time Analysis**: Instant toxicity detection as posts load
- **Nested Comments**: Analyzes entire conversation threads
- **Confidence Scores**: Shows prediction probability for transparency
- **Infinite Scroll**: Load additional posts dynamically
- **Visual Indicators**: Color-coded badges (red for toxic, green for non-toxic)
- **Hover Tooltips**: Detailed prediction metrics on hover

---

## Technical Details

### APIs

**Model API (Port 8001)**
- Accepts text arrays for batch prediction
- Returns toxicity labels with confidence scores
- Caches model in memory for fast inference

**Data API (Port 8000)**
- Fetches Reddit posts via JSON endpoints
- Supports pagination with offset-based loading
- Preprocesses text (removes URLs, mentions, etc.)

### Frontend
- Pure HTML/CSS/JavaScript (no frameworks)
- Responsive design
- Real-time API integration

---

**Model Not Loading:**
- Ensure the checkpoint directory exists at `Hate_Detection_Model/train_transformer_h_files/checkpoint-20160/`
- Check that all model files (config.json, pytorch_model.bin, etc.) are present

**CUDA/GPU Errors:**
- The model will automatically fall back to CPU if CUDA is unavailable
- For CPU-only environments, install PyTorch without CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## License

This project is for educational and research purposes. Please ensure compliance with Reddit's API Terms of Service and the HateXplain dataset license.

---

## Contact

For questions, issues, or contributions, please open an issue on the project repository.
